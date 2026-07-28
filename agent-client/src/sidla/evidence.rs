//! Validation harness. Fork-only: measures the protocol's claims and writes a
//! dated, reproducible report to `doc/evidence/sidla/`.
//!
//! Not part of the harness itself and not built into the binary — everything
//! here is `#[cfg(test)]`, and both entry points are `#[ignore]` so an ordinary
//! `cargo test` run does not pay for them.
//!
//! ```text
//! cargo test -p agent-client --features sidla --release \
//!     sidla::evidence -- --ignored --nocapture
//! ```
//!
//! The offline measurements need nothing but the repository. The live section
//! is skipped unless `SIDLA_LIVE_API_KEY` is set, and never records the key.

use std::collections::HashSet;
use std::fmt::Write as _;

use serde_json::{json, Value};

use super::encode;
use super::packet::{Act, EntityId, Header, Iff, Loc, Obj, Packet, Sta};
use super::shuffle::mix;
use super::wire::{self, Wire};
use super::{decode, fsm, schema};
use crate::state::SharedState;

const PROTOCOL_BRIEF: &str = include_str!("../../data/prompts/sidla_protocol.txt");

/// Fixed so a third party reproduces the same corpus.
const FUZZ_SEED: u64 = 0x5344_4C41_2026;
/// Randomly generated packets pushed through parse and validation.
const SCHEMA_FUZZ_CASES: usize = 100_000;
/// Well-formed packets, every one of which must be admitted.
const VALID_CORPUS_CASES: usize = 20_000;
/// Valid packets broken in exactly one way, every one of which must be rejected.
const MUTATION_CORPUS_CASES: usize = 5_000;
/// Randomly generated replies pushed through the whole pipeline. Lower than
/// the schema count because each one re-encodes a world.
const PIPELINE_FUZZ_CASES: usize = 5_000;
/// Live provider calls. Deliberately small: the point is whether a real model
/// conforms, which a handful of turns already shows.
const LIVE_TURNS: usize = 10;

const REPORT_DIR: &str = "../doc/evidence/sidla";

// ---------------------------------------------------------------- world setup

fn player(id: u64, name: &str, x: f32, z: f32, official: bool) -> onlinerpg_shared::Player {
    onlinerpg_shared::Player {
        id: onlinerpg_shared::PlayerId::from(id),
        name: name.to_string(),
        position: onlinerpg_shared::Position { x, y: 0.0, z },
        rotation: 0.0,
        level: 5,
        health: 100,
        max_health: 100,
        class: onlinerpg_shared::CharacterClass::Knight,
        gender: Default::default(),
        is_official_npc: official,
        torch_on: false,
        floor_level: 0,
        object_type: None,
        main_hand: None,
        object_id: None,
        last_combat_at: 0,
        client_kind: Default::default(),
    }
}

fn monster(id: &str, x: f32, z: f32, aggressive: bool, health: u32) -> onlinerpg_shared::Monster {
    onlinerpg_shared::Monster {
        id: id.to_string(),
        monster_type: "slime".to_string(),
        position: onlinerpg_shared::Position { x, y: 0.0, z },
        rotation: 0.0,
        state: onlinerpg_shared::MonsterState::Idle,
        owner_id: None,
        health,
        max_health: 20,
        floor_level: 0,
        level_override: None,
        aggressive,
        last_attack_at: 0,
        last_move_at: 0,
        move_budget: 0.0,
    }
}

/// A world with `monsters` hostiles/neutrals and one official NPC. Returns the
/// state plus the identifiers that legitimately exist in it.
fn build_world(
    monsters: usize,
) -> (
    SharedState,
    Vec<String>,
    tokio::sync::mpsc::Receiver<onlinerpg_shared::ClientMessage>,
) {
    let (mut state, rx) = crate::state::tests::test_state();
    let me = player(1, "Mika", 0.0, 0.0, false);
    state.self_player_id = Some(me.id);
    state.self_player = Some(me);

    let mut real = vec!["Mika".to_string()];

    let saori = player(2, "Saori", 3.0, 1.0, true);
    real.push(saori.name.clone());
    state.nearby_players.insert(saori.id, saori);

    for i in 0..monsters {
        let a = i as f32 * 0.9;
        let id = format!("monster_slime_{i:04x}");
        real.push(id.clone());
        state.nearby_monsters.insert(
            id.clone(),
            monster(
                &id,
                a.cos() * 12.0,
                a.sin() * 12.0,
                i % 2 == 0,
                20 - (i as u32 % 18),
            ),
        );
    }
    (state, real, rx)
}

// ------------------------------------------------------------------- fuzzing

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(1);
        mix(self.0)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

/// Values a field might carry, spanning the legitimate domain and every way a
/// model is known to leave it: prose, a float, a null, an out-of-range code.
fn field_values() -> Vec<Value> {
    vec![
        json!(0),
        json!(1),
        json!(2),
        json!(4),
        json!(8),
        json!(-1),
        json!(9),
        json!(999),
        json!(-101),
        json!(101),
        json!(2.5),
        json!("attack"),
        json!("idle"),
        json!("hostile"),
        json!(true),
        json!(null),
        json!([1, 2, 3]),
        json!({}),
    ]
}

fn entity_values(real: &[String]) -> Vec<Value> {
    let mut v: Vec<Value> = real.iter().map(|r| json!(r)).collect();
    v.extend([
        json!("ancient_dragon"),
        json!("Player_1"),
        json!(""),
        json!(12345),
        json!(null),
        json!("monster_slime_ffff"),
    ]);
    v
}

/// A random object over the field space, including invented keys and headers.
fn fuzz_packet(rng: &mut Rng, real: &[String]) -> Value {
    let headers = [
        json!("A"),
        json!("B"),
        json!("C"),
        json!("D"),
        json!("E"),
        json!("PPLI"),
        json!(1),
        json!(null),
    ];
    let keys = [
        "SUB", "TAR", "IFF", "STA", "REL", "ACT", "OBJ", "LOC", "HP", "MSG", "SPELL", "type",
        "thought",
    ];
    let values = field_values();
    let entities = entity_values(real);
    let locs = [
        json!("Trinity_Cafe"),
        json!([1.0, 0.0, 2.0]),
        json!("nowhere"),
        json!(0),
    ];

    let mut obj = serde_json::Map::new();
    obj.insert("H".into(), rng.pick(&headers).clone());
    for _ in 0..rng.below(6) {
        let key = *rng.pick(&keys);
        let value = match key {
            "SUB" | "TAR" => rng.pick(&entities).clone(),
            "LOC" => rng.pick(&locs).clone(),
            "MSG" => json!("Stand back."),
            _ => rng.pick(&values).clone(),
        };
        obj.insert(key.into(), value);
    }
    Value::Object(obj)
}

/// A well-formed packet: correct header, every required field present with a
/// value inside its domain, optional fields sometimes. These must all be
/// admitted — a validator that rejects everything would satisfy the
/// no-false-admission claim trivially, so the complementary direction has to be
/// measured too.
fn valid_packet(rng: &mut Rng, real: &[String]) -> Packet {
    let subject = EntityId::name(&real[0]);
    let other = EntityId::name(rng.pick(&real[1..]));
    let sta = *rng.pick(Sta::ALL);
    let iff = *rng.pick(Iff::ALL);
    let act = *rng.pick(Act::ALL);
    let obj = *rng.pick(Obj::ALL);

    match rng.below(4) {
        0 => {
            let loc = if rng.below(2) == 0 {
                Loc::Zone("Trinity_Cafe".into())
            } else {
                Loc::Coord([
                    rng.below(200) as f32 / 10.0 - 10.0,
                    0.0,
                    rng.below(200) as f32 / 10.0 - 10.0,
                ])
            };
            let packet = Packet::ppli(subject, sta, loc);
            if rng.below(2) == 0 {
                packet.with_hp(rng.below(101) as u8)
            } else {
                packet
            }
        }
        1 => {
            let packet = Packet::track(subject, other, iff);
            if rng.below(2) == 0 {
                packet.with_rel(rng.below(201) as i32 - 100)
            } else {
                packet
            }
        }
        2 => {
            let packet = Packet::engage(subject, other, act);
            if act == Act::Talk && rng.below(2) == 0 {
                packet.with_msg("Hold there.")
            } else {
                packet
            }
        }
        _ => {
            let packet = Packet::mission(subject, obj);
            if rng.below(2) == 0 {
                packet.with_tar(other)
            } else {
                packet
            }
        }
    }
}

/// The single-field corruptions a model realistically makes. Each takes a valid
/// packet's JSON and breaks exactly one thing, so a rejection is attributable.
const MUTATIONS: [&str; 7] = [
    "drop a required field",
    "add a forbidden field",
    "prose in an enum field",
    "affinity out of range",
    "invented field",
    "unrecognised header",
    "dialogue without ACT = Talk",
];

/// Apply `mutation` to a valid packet. Returns `None` when the mutation does
/// not apply to this packet's header, so it is not counted.
fn mutate(rng: &mut Rng, packet: &Packet, mutation: &str) -> Option<Value> {
    use super::packet::Field;
    let mut obj = match serde_json::to_value(packet) {
        Ok(Value::Object(o)) => o,
        _ => return None,
    };

    let of_rule = |want: schema::Rule| {
        Field::ALL
            .into_iter()
            .filter(|f| schema::rule(packet.h, *f) == want)
            .collect::<Vec<_>>()
    };

    match mutation {
        "drop a required field" => {
            let required = of_rule(schema::Rule::Required);
            let field = required[rng.below(required.len())];
            obj.remove(field.as_str())?;
        }
        "add a forbidden field" => {
            let forbidden = of_rule(schema::Rule::Forbidden);
            let field = forbidden[rng.below(forbidden.len())];
            let value = match field {
                Field::Sub | Field::Tar => json!("Saori"),
                Field::Loc => json!([1.0, 0.0, 1.0]),
                Field::Msg => json!("stray words"),
                Field::Rel => json!(-20),
                Field::Hp => json!(50),
                _ => json!(1),
            };
            obj.insert(field.as_str().into(), value);
        }
        "prose in an enum field" => {
            let present = [Field::Iff, Field::Sta, Field::Act, Field::Obj]
                .into_iter()
                .filter(|f| obj.contains_key(f.as_str()))
                .collect::<Vec<_>>();
            if present.is_empty() {
                return None;
            }
            let field = present[rng.below(present.len())];
            obj.insert(field.as_str().into(), json!("attack"));
        }
        "affinity out of range" => {
            if schema::rule(packet.h, Field::Rel) == schema::Rule::Forbidden {
                return None;
            }
            obj.insert(
                "REL".into(),
                json!(if rng.below(2) == 0 { -101 } else { 101 }),
            );
        }
        "invented field" => {
            obj.insert("SPELL".into(), json!("fireball"));
        }
        "unrecognised header" => {
            obj.insert("H".into(), json!("E"));
        }
        "dialogue without ACT = Talk" => {
            if packet.h != Header::C || packet.act == Some(Act::Talk) {
                return None;
            }
            obj.insert("MSG".into(), json!("stray words"));
        }
        _ => return None,
    }
    Some(Value::Object(obj))
}

/// A random reply body: sometimes valid packets, sometimes packets that are
/// wrong in one specific way, sometimes what a model does instead of packets.
fn fuzz_reply(rng: &mut Rng, real: &[String]) -> String {
    match rng.below(12) {
        0 => "I would rather talk to them first.".to_string(),
        1 => String::new(),
        2 => "{".to_string(),
        3 => format!(
            "```json\n{}\n```",
            serde_json::to_string(&valid_packet(rng, real)).unwrap()
        ),
        4 => {
            let a = fuzz_packet(rng, real);
            let b = fuzz_packet(rng, real);
            format!("{a}\n{b}")
        }
        5 => json!({"thought": "hmm", "actions": [{"type": "attack"}]}).to_string(),
        6..=8 => serde_json::to_string(&valid_packet(rng, real)).unwrap(),
        9 => {
            let valid = valid_packet(rng, real);
            let mutation = *rng.pick(&MUTATIONS);
            match mutate(rng, &valid, mutation) {
                Some(broken) => serde_json::to_string(&broken).unwrap(),
                None => serde_json::to_string(&valid).unwrap(),
            }
        }
        _ => serde_json::to_string(&fuzz_packet(rng, real)).unwrap(),
    }
}

// ------------------------------------------------------ pipeline under test

/// The turn, spelled out so the harness can see each stage's outcome. Mirrors
/// `SidlaBackend::send_message` minus the provider call.
fn run_turn(reply: &str, uplink: &encode::Uplink) -> (Option<String>, Value) {
    match schema::parse_frame(reply).and_then(|p| decode::to_envelope(&p, uplink, 0)) {
        Ok(envelope) => (None, envelope),
        Err(violation) => {
            let packet = fsm::decide(uplink);
            schema::validate(&packet).expect("fallback packet must satisfy the schema");
            let envelope = decode::to_envelope(&[packet], uplink, 0)
                .expect("fallback packet must be translatable");
            (Some(violation.to_string()), envelope)
        }
    }
}

/// Identifiers an envelope asks the game to act on.
fn named_targets(envelope: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(actions) = envelope["actions"].as_array() {
        for action in actions {
            for key in ["monster_id", "player", "target"] {
                if let Some(name) = action[key].as_str() {
                    out.push(name.to_string());
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------ measurements

#[derive(Default)]
struct SchemaFuzz {
    cases: usize,
    admitted: usize,
    rejected: usize,
    parse_errors: usize,
    admitted_but_invalid: usize,
    violation_kinds: std::collections::BTreeMap<String, usize>,
}

fn measure_schema_fuzz(real: &[String]) -> SchemaFuzz {
    let mut rng = Rng(FUZZ_SEED);
    let mut out = SchemaFuzz::default();
    for _ in 0..SCHEMA_FUZZ_CASES {
        out.cases += 1;
        let candidate = serde_json::to_string(&fuzz_packet(&mut rng, real)).unwrap();
        match serde_json::from_str::<Packet>(&candidate) {
            Ok(packet) => match schema::validate(&packet) {
                Ok(()) => {
                    out.admitted += 1;
                    // Independent re-check: an admitted packet must satisfy
                    // the matrix on every field, not merely have passed once.
                    if !recheck_matrix(&packet) {
                        out.admitted_but_invalid += 1;
                    }
                }
                Err(v) => {
                    out.rejected += 1;
                    *out.violation_kinds.entry(violation_kind(&v)).or_default() += 1;
                }
            },
            Err(_) => {
                out.rejected += 1;
                out.parse_errors += 1;
                *out.violation_kinds.entry("Malformed".into()).or_default() += 1;
            }
        }
    }
    out
}

fn violation_kind(v: &schema::Violation) -> String {
    match v {
        schema::Violation::MissingRequired { .. } => "MissingRequired",
        schema::Violation::ForbiddenField { .. } => "ForbiddenField",
        schema::Violation::OutOfRange { .. } => "OutOfRange",
        schema::Violation::DanglingPayload { .. } => "DanglingPayload",
        schema::Violation::Malformed(_) => "Malformed",
        schema::Violation::NoCommand => "NoCommand",
    }
    .to_string()
}

/// Re-derive the verdict from the matrix alone, independently of `validate`.
fn recheck_matrix(packet: &Packet) -> bool {
    use super::packet::Field;
    for field in Field::ALL {
        let present = field.is_present(packet);
        match schema::rule(packet.h, field) {
            schema::Rule::Required if !present => return false,
            schema::Rule::Forbidden if present => return false,
            _ => {}
        }
    }
    true
}

#[derive(Default)]
struct ValidCorpus {
    cases: usize,
    admitted: usize,
    falsely_rejected: usize,
    reasons: Vec<String>,
}

/// The complementary direction: well-formed packets must be admitted. Without
/// this, "no invalid packet was admitted" would be satisfied by a validator
/// that rejects everything.
fn measure_valid_corpus(real: &[String]) -> ValidCorpus {
    let mut rng = Rng(FUZZ_SEED ^ 0xA5A5);
    let mut out = ValidCorpus::default();
    for _ in 0..VALID_CORPUS_CASES {
        out.cases += 1;
        let packet = valid_packet(&mut rng, real);
        // Through the wire and back, so the round trip is measured too.
        let encoded = serde_json::to_string(&packet).expect("serialise valid packet");
        match serde_json::from_str::<Packet>(&encoded).map_err(|e| e.to_string()) {
            Ok(parsed) => match schema::validate(&parsed) {
                Ok(()) => out.admitted += 1,
                Err(v) => {
                    out.falsely_rejected += 1;
                    if out.reasons.len() < 8 {
                        out.reasons.push(format!("{v} in `{encoded}`"));
                    }
                }
            },
            Err(e) => {
                out.falsely_rejected += 1;
                if out.reasons.len() < 8 {
                    out.reasons.push(format!("{e} in `{encoded}`"));
                }
            }
        }
    }
    out
}

#[derive(Default)]
struct MutationCorpus {
    /// mutation → (cases, rejected)
    per_kind: std::collections::BTreeMap<String, (usize, usize)>,
}

impl MutationCorpus {
    fn missed(&self) -> usize {
        self.per_kind.values().map(|(c, r)| c - r).sum()
    }
}

/// Single-field corruptions of otherwise valid packets. Every one must be
/// rejected, and because exactly one thing is broken the rejection is
/// attributable to that thing.
fn measure_mutation_corpus(real: &[String]) -> MutationCorpus {
    let mut rng = Rng(FUZZ_SEED ^ 0x3C3C);
    let mut out = MutationCorpus::default();
    for _ in 0..MUTATION_CORPUS_CASES {
        let base = valid_packet(&mut rng, real);
        for mutation in MUTATIONS {
            let Some(broken) = mutate(&mut rng, &base, mutation) else {
                continue;
            };
            let entry = out.per_kind.entry(mutation.to_string()).or_insert((0, 0));
            entry.0 += 1;
            let text = serde_json::to_string(&broken).expect("serialise mutated packet");
            let rejected = match serde_json::from_str::<Packet>(&text) {
                Ok(packet) => schema::validate(&packet).is_err(),
                Err(_) => true,
            };
            if rejected {
                entry.1 += 1;
            }
        }
    }
    out
}

#[derive(Default)]
struct PipelineFuzz {
    cases: usize,
    accepted: usize,
    fell_back: usize,
    turns_lost: usize,
    unreal_targets: usize,
    untyped_actions: usize,
}

fn measure_pipeline_fuzz(uplink: &encode::Uplink, real: &[String]) -> PipelineFuzz {
    let mut rng = Rng(FUZZ_SEED ^ 0xFFFF);
    let real: HashSet<&str> = real.iter().map(String::as_str).collect();
    let mut out = PipelineFuzz::default();

    for _ in 0..PIPELINE_FUZZ_CASES {
        out.cases += 1;
        let reply = fuzz_reply(
            &mut rng,
            &real.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        );
        let (violation, envelope) = run_turn(&reply, uplink);
        if violation.is_some() {
            out.fell_back += 1;
        } else {
            out.accepted += 1;
        }
        match envelope["actions"].as_array() {
            Some(actions) if !actions.is_empty() => {
                for action in actions {
                    if !action["type"].is_string() {
                        out.untyped_actions += 1;
                    }
                }
            }
            _ => out.turns_lost += 1,
        }
        for name in named_targets(&envelope) {
            if !real.contains(name.as_str()) {
                out.unreal_targets += 1;
            }
        }
    }
    out
}

struct Determinism {
    encodings: usize,
    distinct_frames: usize,
    decodes: usize,
    distinct_envelopes: usize,
    fallbacks: usize,
    distinct_fallbacks: usize,
}

fn measure_determinism(state: &SharedState) -> Determinism {
    const N: usize = 1_000;
    let frames: HashSet<String> = (0..N)
        .map(|_| wire::render_compact(&encode::encode(state).packets))
        .collect();

    let uplink = encode::encode(state);
    let accepted = format!(
        r#"{{"H":"C","SUB":"Mika","TAR":"{}","ACT":2}}"#,
        uplink
            .hostiles()
            .next()
            .map(|t| t.id.to_string())
            .unwrap_or_default()
    );
    let envelopes: HashSet<String> = (0..N)
        .map(|_| run_turn(&accepted, &uplink).1.to_string())
        .collect();
    let fallbacks: HashSet<String> = (0..N)
        .map(|_| run_turn("no packets here", &uplink).1.to_string())
        .collect();

    Determinism {
        encodings: N,
        distinct_frames: frames.len(),
        decodes: N,
        distinct_envelopes: envelopes.len(),
        fallbacks: N,
        distinct_fallbacks: fallbacks.len(),
    }
}

struct TokenRow {
    entities: usize,
    prose: usize,
    json: usize,
    compact: usize,
}

fn measure_tokens() -> Vec<TokenRow> {
    [1usize, 4, 16, 64, 256]
        .into_iter()
        .map(|n| {
            let (state, _real, _rx) = build_world(n);
            let uplink = encode::encode(&state);
            TokenRow {
                entities: uplink.tracks.len(),
                prose: est(&state.format_world_state()),
                json: est(&wire::render_json(&uplink.packets)),
                compact: est(&wire::render_compact(&uplink.packets)),
            }
        })
        .collect()
}

fn est(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Downlink cost: the envelope a prose-answering agent must produce against the
/// packet that replaces it.
fn measure_downlink() -> (usize, usize) {
    let envelope = json!({
        "thought": "A slime is close and I still have most of my health, so I will engage \
                    it before it reaches Saori.",
        "actions": [{"type": "attack", "monster_id": "monster_slime_0003"}],
        "memory_update": "Fought a slime near the cafe while Saori was nearby."
    })
    .to_string();
    let packet = Packet::engage(
        EntityId::name("Mika"),
        EntityId::name("monster_slime_0003"),
        Act::Attack,
    );
    (est(&envelope), est(&wire::render_json(&[packet])))
}

// ------------------------------------------------------------- live provider

struct LiveTurn {
    turn: usize,
    world: String,
    reply: String,
    violation: Option<String>,
    action: String,
    unreal_target: bool,
    subject_dead: bool,
    respawned: bool,
    input_tokens: u64,
    output_tokens: u64,
}

/// One call against the Gemini Interactions API. The key comes from the
/// environment and is never written to the report.
async fn call_gemini(
    api_key: &str,
    model: &str,
    prompt: &str,
) -> anyhow::Result<(String, u64, u64)> {
    let body = json!({
        "model": model,
        "input": prompt,
        "generation_config": { "max_output_tokens": 4096, "thinking_level": "high" }
    });
    let response = reqwest::Client::new()
        .post("https://generativelanguage.googleapis.com/v1beta/interactions")
        .header("x-goog-api-key", api_key)
        .header("Api-Revision", "2026-05-20")
        .json(&body)
        .send()
        .await?;
    let status = response.status();
    let payload: Value = response.json().await?;
    if !status.is_success() {
        anyhow::bail!("provider returned {status}: {payload}");
    }
    let text = payload["steps"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|step| step["type"] == "model_output")
        .flat_map(|step| step["content"].as_array().into_iter().flatten())
        .filter(|part| part["type"] == "text")
        .filter_map(|part| part["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let usage = &payload["usage"];
    Ok((
        text,
        usage["total_input_tokens"].as_u64().unwrap_or(0),
        usage["total_output_tokens"].as_u64().unwrap_or(0),
    ))
}

fn live_prompt(uplink: &encode::Uplink) -> String {
    let mut prompt = String::from(PROTOCOL_BRIEF);
    prompt.push_str("\n=== FRAME ===\n");
    prompt.push_str(&wire::render(&uplink.packets, Wire::Compact));
    prompt.push_str("\n\nReply with SIDLA packets.");
    prompt
}

async fn measure_live(api_key: &str, model: &str) -> Vec<LiveTurn> {
    let mut out = Vec::new();
    for turn in 0..LIVE_TURNS {
        let monsters = [0usize, 1, 2, 3, 5, 8, 1, 2, 4, 6][turn % 10];
        let (mut state, real, _rx) = build_world(monsters);
        if turn % 3 == 1 {
            state.self_player.as_mut().unwrap().health = 12;
        }
        if turn % 5 == 4 {
            state.self_player.as_mut().unwrap().health = 0;
        }
        let uplink = encode::encode(&state);
        let world = format!(
            "monsters={monsters} self_hp={}% sta={:?}",
            uplink.subject_hp_pct, uplink.subject_sta
        );

        let (reply, input_tokens, output_tokens) =
            match call_gemini(api_key, model, &live_prompt(&uplink)).await {
                Ok(v) => v,
                Err(e) => {
                    out.push(LiveTurn {
                        turn,
                        world,
                        reply: format!("[provider error] {e}"),
                        violation: Some("provider error".into()),
                        action: "n/a".into(),
                        unreal_target: false,
                        subject_dead: false,
                        respawned: false,
                        input_tokens: 0,
                        output_tokens: 0,
                    });
                    continue;
                }
            };

        let (violation, envelope) = run_turn(&reply, &uplink);
        let real: HashSet<&str> = real.iter().map(String::as_str).collect();
        let unreal_target = named_targets(&envelope)
            .iter()
            .any(|n| !real.contains(n.as_str()));

        let respawned = envelope["actions"]
            .as_array()
            .is_some_and(|a| a.iter().any(|x| x["type"] == "respawn"));

        out.push(LiveTurn {
            turn,
            world,
            reply: reply.trim().to_string(),
            violation,
            action: envelope["actions"].to_string(),
            unreal_target,
            subject_dead: uplink.subject_sta == Sta::Dead,
            respawned,
            input_tokens,
            output_tokens,
        });
    }
    out
}

// ---------------------------------------------------------------- reporting

/// Which claims this implementation can and cannot speak to. Kept beside the
/// measurements so the report never implies coverage it does not have.
const CLAIM_MAP: &[(&str, &str, &str)] = &[
    (
        "Header selection and structured packet assembly",
        "Implemented",
        "packet.rs, wire.rs, encode.rs",
    ),
    (
        "Masking control: required/forbidden field verification",
        "Implemented",
        "schema.rs `rule`, `validate`",
    ),
    (
        "Discard on violation, fall back to default/FSM control",
        "Implemented",
        "fsm.rs `decide`, backend.rs `fallback`",
    ),
    (
        "Enum/hash-only decision fields, no natural language",
        "Implemented",
        "packet.rs `code_enum`, integer-only deserialisation",
    ),
    (
        "Four functional header types A/B/C/D",
        "Implemented",
        "packet.rs `Header`, schema.rs matrix",
    ),
    (
        "Integer shuffling outside the inference engine",
        "Implemented (partial)",
        "shuffle.rs `vary`, equivalence classes only",
    ),
    (
        "State encoder to deterministic latent coordinates (DGAE-WL)",
        "Not implemented",
        "external provider; see README",
    ),
    (
        "Sub-2-bit quantised inference engine, binary/ternary modes",
        "Not implemented",
        "external provider; see README",
    ),
    (
        "Temperature = 0.0 greedy decoding",
        "Not implemented",
        "provider-side setting, not controlled here",
    ),
];

struct Report {
    commit: String,
    rustc: String,
    os: String,
    timestamp: String,
    schema: SchemaFuzz,
    valid: ValidCorpus,
    mutations: MutationCorpus,
    pipeline: PipelineFuzz,
    determinism: Determinism,
    tokens: Vec<TokenRow>,
    downlink: (usize, usize),
    live: Vec<LiveTurn>,
    live_model: Option<String>,
}

fn shell(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

impl Report {
    fn markdown(&self) -> String {
        let mut m = String::new();
        let _ = writeln!(m, "# SIDLA validation report\n");
        let _ = writeln!(
            m,
            "Generated by `agent-client/src/sidla/evidence.rs`. Reproduce with:\n"
        );
        let _ = writeln!(
            m,
            "```\ncargo test -p agent-client --features sidla --release \\\n    \
             sidla::evidence -- --ignored --nocapture\n```\n"
        );

        let _ = writeln!(m, "## Environment\n");
        let _ = writeln!(m, "| Item | Value |");
        let _ = writeln!(m, "| :--- | :--- |");
        let _ = writeln!(m, "| Timestamp (UTC) | {} |", self.timestamp);
        let _ = writeln!(m, "| Commit | `{}` |", self.commit);
        let _ = writeln!(m, "| Toolchain | {} |", self.rustc);
        let _ = writeln!(m, "| OS | {} |", self.os);
        let _ = writeln!(m, "| Fuzz seed | `{FUZZ_SEED:#x}` (fixed) |");
        let _ = writeln!(
            m,
            "| Live model | {} |\n",
            self.live_model.as_deref().unwrap_or("not run")
        );

        let _ = writeln!(m, "## Claim coverage\n");
        let _ = writeln!(m, "| Element | Status | Implementation |");
        let _ = writeln!(m, "| :--- | :--- | :--- |");
        for (element, status, where_) in CLAIM_MAP {
            let _ = writeln!(m, "| {element} | {status} | `{where_}` |");
        }
        let _ = writeln!(
            m,
            "\nThe three unimplemented elements concern the inference engine. \
             This client calls an external provider, so they are out of reach \
             here and no measurement below speaks to them.\n"
        );

        let v = &self.valid;
        let _ = writeln!(m, "## 1. Well-formed packets are admitted\n");
        let _ = writeln!(
            m,
            "The complementary direction, measured first because without it the \
             rejection results below would be satisfied by a validator that \
             refuses everything. Randomly generated packets, correct by \
             construction, serialised and read back.\n"
        );
        let _ = writeln!(m, "| Measurement | Result |");
        let _ = writeln!(m, "| :--- | ---: |");
        let _ = writeln!(m, "| Cases | {} |", v.cases);
        let _ = writeln!(
            m,
            "| Admitted | {} ({:.2} %) |",
            v.admitted,
            pct(v.admitted, v.cases)
        );
        let _ = writeln!(m, "| **Falsely rejected** | **{}** |\n", v.falsely_rejected);
        if !v.reasons.is_empty() {
            let _ = writeln!(m, "Sample false rejections:\n");
            for reason in &v.reasons {
                let _ = writeln!(m, "- `{reason}`");
            }
            let _ = writeln!(m);
        }

        let mu = &self.mutations;
        let _ = writeln!(m, "## 2. Single-field corruptions are rejected\n");
        let _ = writeln!(
            m,
            "Valid packets broken in exactly one way. Because only one thing is \
             wrong, each rejection is attributable to that thing rather than to \
             the reply being unreadable.\n"
        );
        let _ = writeln!(m, "| Corruption | Cases | Rejected | Rate |");
        let _ = writeln!(m, "| :--- | ---: | ---: | ---: |");
        for (kind, (cases, rejected)) in &mu.per_kind {
            let _ = writeln!(
                m,
                "| {kind} | {cases} | {rejected} | {:.2} % |",
                pct(*rejected, *cases)
            );
        }
        let _ = writeln!(
            m,
            "\n**Corruptions that slipped through: {}**\n",
            mu.missed()
        );

        let s = &self.schema;
        let _ = writeln!(m, "## 3. Random-object fuzzing\n");
        let _ = writeln!(
            m,
            "Randomly generated packet objects over the full field space, \
             including invented keys and headers, prose in enum positions, \
             floats, nulls and out-of-range codes.\n"
        );
        let _ = writeln!(m, "| Measurement | Result |");
        let _ = writeln!(m, "| :--- | ---: |");
        let _ = writeln!(m, "| Cases | {} |", s.cases);
        let _ = writeln!(
            m,
            "| Admitted | {} ({:.2} %) |",
            s.admitted,
            pct(s.admitted, s.cases)
        );
        let _ = writeln!(
            m,
            "| Rejected | {} ({:.2} %) |",
            s.rejected,
            pct(s.rejected, s.cases)
        );
        let _ = writeln!(m, "| — of which failed to parse | {} |", s.parse_errors);
        let _ = writeln!(
            m,
            "| **Admitted yet violating the matrix** | **{}** |\n",
            s.admitted_but_invalid
        );
        let _ = writeln!(m, "Rejection reasons:\n");
        let _ = writeln!(m, "| Reason | Count |");
        let _ = writeln!(m, "| :--- | ---: |");
        for (kind, count) in &s.violation_kinds {
            let _ = writeln!(m, "| {kind} | {count} |");
        }
        let _ = writeln!(m);

        let p = &self.pipeline;
        let _ = writeln!(m, "## 4. End-to-end fuzzing\n");
        let _ = writeln!(
            m,
            "Randomly generated replies — packets, prose, truncated JSON, \
             fenced blocks — pushed through encode, validate, decode and \
             fallback against a fixed world.\n"
        );
        let _ = writeln!(m, "| Measurement | Result |");
        let _ = writeln!(m, "| :--- | ---: |");
        let _ = writeln!(m, "| Cases | {} |", p.cases);
        let _ = writeln!(
            m,
            "| Accepted from the reply | {} ({:.2} %) |",
            p.accepted,
            pct(p.accepted, p.cases)
        );
        let _ = writeln!(
            m,
            "| Answered by the fallback | {} ({:.2} %) |",
            p.fell_back,
            pct(p.fell_back, p.cases)
        );
        let _ = writeln!(
            m,
            "| **Turns lost (no action reached the game)** | **{}** |",
            p.turns_lost
        );
        let _ = writeln!(
            m,
            "| **Commands naming an entity not in the world** | **{}** |",
            p.unreal_targets
        );
        let _ = writeln!(
            m,
            "| **Actions without a type** | **{}** |\n",
            p.untyped_actions
        );

        let d = &self.determinism;
        let _ = writeln!(m, "## 5. Determinism\n");
        let _ = writeln!(m, "| Measurement | Repetitions | Distinct results |");
        let _ = writeln!(m, "| :--- | ---: | ---: |");
        let _ = writeln!(
            m,
            "| Uplink frame from one world | {} | {} |",
            d.encodings, d.distinct_frames
        );
        let _ = writeln!(
            m,
            "| Envelope from one accepted reply | {} | {} |",
            d.decodes, d.distinct_envelopes
        );
        let _ = writeln!(
            m,
            "| Envelope from one rejected reply | {} | {} |\n",
            d.fallbacks, d.distinct_fallbacks
        );

        let _ = writeln!(m, "## 6. Token cost\n");
        let _ = writeln!(
            m,
            "Estimated at four characters per token, used only to compare \
             renderings of the same content.\n"
        );
        let _ = writeln!(
            m,
            "| Entities in sight | Prose state | SIDLA json | SIDLA compact | compact vs prose |"
        );
        let _ = writeln!(m, "| ---: | ---: | ---: | ---: | ---: |");
        for row in &self.tokens {
            let delta = (row.compact as f64 / row.prose as f64 - 1.0) * 100.0;
            let _ = writeln!(
                m,
                "| {} | {} | {} | {} | {delta:+.0} % |",
                row.entities, row.prose, row.json, row.compact
            );
        }
        let (envelope, packet) = self.downlink;
        let _ = writeln!(
            m,
            "\nDownlink, one decision: prose envelope {envelope} tokens against \
             a packet at {packet} tokens ({:+.0} %).\n",
            (packet as f64 / envelope as f64 - 1.0) * 100.0
        );
        let _ = writeln!(
            m,
            "The uplink saving is modest and narrows as the world fills, because \
             the prose it replaces is already terse; the json rendering is dearer \
             than prose outright. The saving is on the downlink.\n"
        );

        let _ = writeln!(m, "## 7. Live provider\n");
        if self.live.is_empty() {
            let _ = writeln!(
                m,
                "Not run. Set `SIDLA_LIVE_API_KEY` to include this section.\n"
            );
        } else {
            let violations = self.live.iter().filter(|t| t.violation.is_some()).count();
            let unreal = self.live.iter().filter(|t| t.unreal_target).count();
            let _ = writeln!(
                m,
                "{} turns against a real model, each given the protocol brief and \
                 a compact frame.\n",
                self.live.len()
            );
            let _ = writeln!(m, "| Measurement | Result |");
            let _ = writeln!(m, "| :--- | ---: |");
            let _ = writeln!(m, "| Turns | {} |", self.live.len());
            let _ = writeln!(
                m,
                "| Replies rejected by validation | {violations} ({:.0} %) |",
                pct(violations, self.live.len())
            );
            let _ = writeln!(
                m,
                "| **Commands naming an entity not in the world** | **{unreal}** |"
            );
            let _ = writeln!(
                m,
                "| **Turns lost** | **{}** |\n",
                self.live
                    .iter()
                    .filter(|t| t.action == "[]" || t.action == "n/a")
                    .count()
            );

            let dead_turns: Vec<&LiveTurn> = self.live.iter().filter(|t| t.subject_dead).collect();
            let dead_unrecovered = dead_turns
                .iter()
                .filter(|t| t.violation.is_none() && !t.respawned)
                .count();
            let dead_recovered_by_fallback = dead_turns
                .iter()
                .filter(|t| t.violation.is_some() && t.respawned)
                .count();
            if !dead_turns.is_empty() {
                let _ = writeln!(m, "### Observation: conformance is not quality\n");
                let _ = writeln!(
                    m,
                    "Of {} turns where the agent was dead, {} produced an admitted \
                     reply that nonetheless failed to respawn, and {} were \
                     recovered by the fallback after the reply was rejected. \
                     Validation guarantees that a command is structurally sound \
                     and refers to a real entity; it does not guarantee the \
                     decision is a good one. In this sample the fallback made the \
                     better choice in the situation the model handled worst.\n",
                    dead_turns.len(),
                    dead_unrecovered,
                    dead_recovered_by_fallback
                );
            }

            let _ = writeln!(m, "### Per-turn detail\n");
            for t in &self.live {
                let _ = writeln!(m, "#### Turn {} — {}\n", t.turn, t.world);
                let _ = writeln!(
                    m,
                    "Provider tokens: {} in, {} out.\n",
                    t.input_tokens, t.output_tokens
                );
                let _ = writeln!(m, "Reply:\n\n```\n{}\n```\n", t.reply);
                match &t.violation {
                    Some(v) => {
                        let _ = writeln!(m, "Verdict: **rejected** — {v}\n");
                    }
                    None => {
                        let _ = writeln!(m, "Verdict: **admitted**\n");
                    }
                }
                let _ = writeln!(m, "Reached the game:\n\n```json\n{}\n```\n", t.action);
            }
        }

        let _ = writeln!(m, "## Limitations\n");
        let _ = writeln!(
            m,
            "- The token figures are character-based estimates, not a tokeniser's \
             output. They compare two renderings and are not absolute costs.\n\
             - The live sample is {} turns against one model. It shows whether a \
             real model conforms; it does not establish a rate.\n\
             - The fuzz corpus is generated from a hand-written value space. It \
             covers the failure modes the authors anticipated, not all possible \
             output.\n\
             - Determinism is measured within one process on one machine. Float \
             formatting is fixed to one decimal, but cross-platform identity is \
             not tested here.\n\
             - Three claimed elements are unimplemented (see Claim coverage) and \
             nothing here bears on them.",
            self.live.len()
        );
        m
    }

    fn json(&self) -> Value {
        json!({
            "environment": {
                "timestamp_utc": self.timestamp,
                "commit": self.commit,
                "toolchain": self.rustc,
                "os": self.os,
                "fuzz_seed": format!("{FUZZ_SEED:#x}"),
                "live_model": self.live_model,
            },
            "claim_coverage": CLAIM_MAP.iter().map(|(e, s, w)| json!({
                "element": e, "status": s, "implementation": w
            })).collect::<Vec<_>>(),
            "valid_corpus": {
                "cases": self.valid.cases,
                "admitted": self.valid.admitted,
                "falsely_rejected": self.valid.falsely_rejected,
            },
            "mutation_corpus": {
                "per_kind": self.mutations.per_kind.iter().map(|(k, (c, r))| json!({
                    "corruption": k, "cases": c, "rejected": r
                })).collect::<Vec<_>>(),
                "missed": self.mutations.missed(),
            },
            "schema_fuzz": {
                "cases": self.schema.cases,
                "admitted": self.schema.admitted,
                "rejected": self.schema.rejected,
                "parse_errors": self.schema.parse_errors,
                "admitted_but_invalid": self.schema.admitted_but_invalid,
                "violation_kinds": self.schema.violation_kinds,
            },
            "pipeline_fuzz": {
                "cases": self.pipeline.cases,
                "accepted": self.pipeline.accepted,
                "fell_back": self.pipeline.fell_back,
                "turns_lost": self.pipeline.turns_lost,
                "unreal_targets": self.pipeline.unreal_targets,
                "untyped_actions": self.pipeline.untyped_actions,
            },
            "determinism": {
                "uplink_repetitions": self.determinism.encodings,
                "uplink_distinct": self.determinism.distinct_frames,
                "accepted_repetitions": self.determinism.decodes,
                "accepted_distinct": self.determinism.distinct_envelopes,
                "fallback_repetitions": self.determinism.fallbacks,
                "fallback_distinct": self.determinism.distinct_fallbacks,
            },
            "tokens": {
                "uplink": self.tokens.iter().map(|r| json!({
                    "entities": r.entities, "prose": r.prose,
                    "sidla_json": r.json, "sidla_compact": r.compact
                })).collect::<Vec<_>>(),
                "downlink": { "prose_envelope": self.downlink.0, "packet": self.downlink.1 },
            },
            "live": self.live.iter().map(|t| json!({
                "turn": t.turn,
                "world": t.world,
                "reply": t.reply,
                "violation": t.violation,
                "reached_game": t.action,
                "unreal_target": t.unreal_target,
                "subject_dead": t.subject_dead,
                "respawned": t.respawned,
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
            })).collect::<Vec<_>>(),
        })
    }
}

fn pct(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        return 0.0;
    }
    part as f64 / whole as f64 * 100.0
}

fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let (h, mi, s) = ((secs % 86_400) / 3600, (secs % 3600) / 60, secs % 60);
    let (y, mo, d) = civil_from_days(days as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Days since the Unix epoch to a calendar date (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ------------------------------------------------------------- entry points

#[tokio::test]
#[ignore = "measurement harness: writes a report, and optionally calls a provider"]
async fn generate_validation_report() {
    let (state, real, _rx) = build_world(4);
    let uplink = encode::encode(&state);

    println!("SIDLA validation: valid corpus ({VALID_CORPUS_CASES} cases)...");
    let valid = measure_valid_corpus(&real);
    println!("SIDLA validation: mutation corpus ({MUTATION_CORPUS_CASES} bases)...");
    let mutations = measure_mutation_corpus(&real);
    println!("SIDLA validation: schema fuzzing ({SCHEMA_FUZZ_CASES} cases)...");
    let schema_fuzz = measure_schema_fuzz(&real);
    println!("SIDLA validation: pipeline fuzzing ({PIPELINE_FUZZ_CASES} cases)...");
    let pipeline = measure_pipeline_fuzz(&uplink, &real);
    println!("SIDLA validation: determinism...");
    let determinism = measure_determinism(&state);
    println!("SIDLA validation: token cost...");
    let tokens = measure_tokens();
    let downlink = measure_downlink();

    let api_key = std::env::var("SIDLA_LIVE_API_KEY")
        .ok()
        .filter(|k| !k.is_empty());
    let model =
        std::env::var("SIDLA_LIVE_MODEL").unwrap_or_else(|_| "gemini-3.6-flash".to_string());
    let live = match api_key.as_deref() {
        Some(key) => {
            println!("SIDLA validation: {LIVE_TURNS} live turns against {model}...");
            measure_live(key, &model).await
        }
        None => {
            println!("SIDLA validation: live section skipped (SIDLA_LIVE_API_KEY unset)");
            Vec::new()
        }
    };

    let report = Report {
        commit: shell("git", &["rev-parse", "HEAD"]),
        rustc: shell("rustc", &["--version"]),
        os: format!("{} {}", shell("uname", &["-s"]), shell("uname", &["-r"])),
        timestamp: utc_now(),
        schema: schema_fuzz,
        valid,
        mutations,
        pipeline,
        determinism,
        tokens,
        downlink,
        live_model: api_key.as_ref().map(|_| model.clone()),
        live,
    };

    // The claims, asserted rather than merely reported: a report that records a
    // leak should also fail the run.
    assert_eq!(
        report.schema.admitted_but_invalid, 0,
        "a packet was admitted that violates the field matrix"
    );
    assert_eq!(
        report.valid.falsely_rejected, 0,
        "a well-formed packet was rejected: {:?}",
        report.valid.reasons
    );
    assert!(
        report.valid.admitted > 0,
        "the valid corpus produced nothing"
    );
    assert_eq!(
        report.mutations.missed(),
        0,
        "a single-field corruption was admitted"
    );
    for (kind, (cases, _)) in &report.mutations.per_kind {
        assert!(*cases > 0, "mutation `{kind}` was never exercised");
    }
    assert_eq!(
        report.pipeline.unreal_targets, 0,
        "a command reached the game naming an entity not in the world"
    );
    assert_eq!(report.pipeline.turns_lost, 0, "a turn produced no action");
    assert_eq!(report.pipeline.untyped_actions, 0, "an action had no type");
    assert_eq!(report.determinism.distinct_frames, 1);
    assert_eq!(report.determinism.distinct_envelopes, 1);
    assert_eq!(report.determinism.distinct_fallbacks, 1);
    for turn in &report.live {
        assert!(
            !turn.unreal_target,
            "live turn {} reached the game with an unreal target",
            turn.turn
        );
    }

    std::fs::create_dir_all(REPORT_DIR).expect("create report directory");
    let date = report.timestamp.split('T').next().unwrap_or("undated");
    let stem = format!("{REPORT_DIR}/report_{date}");
    std::fs::write(format!("{stem}.md"), report.markdown()).expect("write markdown report");
    std::fs::write(
        format!("{stem}.json"),
        serde_json::to_string_pretty(&report.json()).expect("serialise report"),
    )
    .expect("write json report");

    println!("\n{}", report.markdown());
    println!("Report written to {stem}.md and {stem}.json");
}

/// The corpus generators must actually produce the shapes they claim to, or the
/// headline counts would be measuring nothing.
#[test]
fn the_fuzz_corpus_covers_its_intended_failure_modes() {
    let (_state, real, _rx) = build_world(2);
    let mut rng = Rng(FUZZ_SEED);
    let mut kinds = std::collections::BTreeSet::new();
    let mut parsed = 0;
    let mut admitted = 0;

    for _ in 0..20_000 {
        let candidate = serde_json::to_string(&fuzz_packet(&mut rng, &real)).unwrap();
        match serde_json::from_str::<Packet>(&candidate) {
            Ok(packet) => {
                parsed += 1;
                match schema::validate(&packet) {
                    Ok(()) => admitted += 1,
                    Err(v) => {
                        kinds.insert(violation_kind(&v));
                    }
                }
            }
            Err(_) => {
                kinds.insert("Malformed".into());
            }
        }
    }

    assert!(parsed > 0, "corpus never produced a parseable packet");
    assert!(admitted > 0, "corpus never produced a valid packet");
    for expected in ["Malformed", "MissingRequired", "ForbiddenField"] {
        assert!(
            kinds.contains(expected),
            "corpus never produced {expected}; got {kinds:?}"
        );
    }
}

#[test]
fn the_report_renders_without_a_live_section() {
    let (state, real, _rx) = build_world(2);
    let uplink = encode::encode(&state);
    let report = Report {
        commit: "test".into(),
        rustc: "test".into(),
        os: "test".into(),
        timestamp: utc_now(),
        schema: SchemaFuzz::default(),
        valid: ValidCorpus::default(),
        mutations: MutationCorpus::default(),
        pipeline: measure_pipeline_fuzz(&uplink, &real),
        determinism: measure_determinism(&state),
        tokens: vec![],
        downlink: measure_downlink(),
        live: vec![],
        live_model: None,
    };
    let markdown = report.markdown();
    assert!(markdown.contains("Not run."));
    assert!(markdown.contains("Claim coverage"));
    assert!(markdown.contains("Not implemented"));
    assert!(report.json()["live"].as_array().unwrap().is_empty());
}

#[test]
fn the_timestamp_helper_agrees_with_known_dates() {
    assert_eq!(civil_from_days(0), (1970, 1, 1));
    assert_eq!(civil_from_days(19_723), (2024, 1, 1));
    assert_eq!(civil_from_days(20_636), (2026, 7, 2));
}

#[test]
fn unimplemented_claims_are_declared_as_such() {
    let unimplemented = CLAIM_MAP
        .iter()
        .filter(|(_, status, _)| status.starts_with("Not implemented"))
        .count();
    assert_eq!(
        unimplemented, 3,
        "claim coverage table lost an honest entry"
    );
}
