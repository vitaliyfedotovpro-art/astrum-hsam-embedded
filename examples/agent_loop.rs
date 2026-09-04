//! What using this engine actually looks like in your own code.
//!
//! Not a benchmark. The other three examples measure things; this one answers the question
//! a reader asks first — "what does my code look like if I use this?" — end to end: facts in
//! with their provenance, a question, the recalled set, and the prompt block you hand to an
//! LLM.
//!
//! Two honest caveats about what is NOT here, because they are the whole reason this file is
//! short:
//!
//! 1. **There is no embedder.** This engine never turns text into vectors; you bring them
//!    from your NPU, your host, or an embedding API. So that the example stays deterministic
//!    and offline, the vectors below are written by hand over four made-up axes
//!    (probe care / scheduling / billing / self-reference). That is a stand-in, not a
//!    suggestion — with a real embedder these come from the text.
//! 2. **There is no LLM call.** The last step prints the prompt block rather than sending it.
//!    Where it goes and how you frame it is your business; what this engine decides is what
//!    is allowed into that block in the first place.
//!
//! Run: cargo run --release --example agent_loop

use astrum_memory::{CanonLevel, MemoryGraphNexus, SimpleVectorIndex, SourceType};

/// The ranking the C ABI applies, reproduced here so the example shows the real rule rather
/// than a simplified one: drop anything below the relevance floor, drop quarantined sources
/// outright (a rank multiplier of zero means excluded, not merely demoted), then order by
/// cosine weighted by provenance.
const RELEVANCE_FLOOR: f32 = 0.25;

fn main() {
    let mut nexus = MemoryGraphNexus::new();
    let mut index = SimpleVectorIndex::new(4);

    // ── 1. Give it facts, and say where each one came from ───────────────────────────────
    //
    // Provenance is not metadata you attach for later auditing. It decides what may be
    // recalled at all, and it is the one field you cannot infer after the fact — only the
    // code that ingests knows whether this text came from a person, a manual, or the model.

    let add = |nexus: &mut MemoryGraphNexus,
               index: &mut SimpleVectorIndex,
               content: &str,
               source: SourceType,
               canon: CanonLevel,
               embedding: [f32; 4]| {
        let id = nexus.create_node(
            content.into(),
            String::new(),
            Vec::new(),
            source,
            2,
            None,
            canon,
        );
        index.insert(id.clone(), embedding.to_vec()).unwrap();
        id
    };

    // The operator's own words, and a rule you cannot afford to lose: marked canon, which
    // makes it exempt from eviction however rarely it is read.
    let protocol = add(
        &mut nexus,
        &mut index,
        "Probes are disinfected for 12 minutes in the OPA bath. Never autoclave them.",
        SourceType::UserUtterance,
        CanonLevel::L2Foundational,
        [1.0, 0.0, 0.0, 0.0],
    );

    // A manual. Trusted, but not the operator.
    let manual = add(
        &mut nexus,
        &mut index,
        "Manufacturer manual: autoclaving the probe voids the warranty.",
        SourceType::ExternalDoc,
        CanonLevel::None,
        [0.92, 0.0, 0.0, 0.10],
    );

    // Something the model produced earlier. Plausible, close to the question, and wrong by
    // two minutes. It stays recallable, at a lower weight.
    add(
        &mut nexus,
        &mut index,
        "Most clinics disinfect probes for about 10 minutes.",
        SourceType::LlmGeneration,
        CanonLevel::None,
        [0.88, 0.0, 0.0, 0.05],
    );

    // The model describing itself. This is the one that poisons a plain vector store: it is
    // topically close to almost any question about the domain, so it keeps surfacing, and
    // once it surfaces it reads like a fact the assistant knows.
    add(
        &mut nexus,
        &mut index,
        "I am an assistant for clinical protocols and I always follow disinfection guidance.",
        SourceType::LlmSelfDescription,
        CanonLevel::None,
        [0.80, 0.0, 0.0, 0.55],
    );

    // Unrelated to the question by embedding, but part of the same procedure in practice.
    // Nothing about the wording of the question will retrieve it; the edge below will.
    let bath_log = add(
        &mut nexus,
        &mut index,
        "The OPA bath is replaced every 14 days; log the date on the bottle.",
        SourceType::ExternalDoc,
        CanonLevel::None,
        [0.05, 0.0, 0.95, 0.0],
    );

    add(
        &mut nexus,
        &mut index,
        "Clinic hours are 8 to 6, Tuesday through Saturday.",
        SourceType::UserUtterance,
        CanonLevel::None,
        [0.0, 1.0, 0.0, 0.0],
    );

    // ── 2. Say what belongs with what ────────────────────────────────────────────────────
    //
    // Edges are how a fact becomes reachable from a question that does not resemble it.
    // You build them: from co-occurrence in a session, from a shared entity, from an
    // explicit link. The engine does not invent them.
    nexus.add_edge(&protocol, &manual, "same_procedure", 0.9);
    nexus.add_edge(&protocol, &bath_log, "same_procedure", 0.8);

    // ── 3. Ask something ─────────────────────────────────────────────────────────────────

    let question = "how long do I disinfect the probe?";
    let query = [1.0f32, 0.0, 0.0, 0.0]; // your embedder's output for `question`

    println!("Q: {question}\n");

    // What a plain vector store would hand you. Note what is in it.
    println!("Flat cosine, top 4 — what a vector store returns:");
    for hit in index.search(&query, 4) {
        let node = &nexus.get_node()[&hit.node_id];
        println!(
            "  {:.3}  [{:?}]  {}",
            hit.score, node.source_type, node.content
        );
    }

    // The same query through the provenance gate.
    println!("\nProvenance-gated recall — what this engine returns:");
    let mut ranked: Vec<(f32, f32, &str, SourceType)> = index
        .search(&query, 8)
        .into_iter()
        .filter(|hit| hit.score >= RELEVANCE_FLOOR)
        .filter_map(|hit| {
            let node = &nexus.get_node()[&hit.node_id];
            let weight = node.source_type.rank_multiplier();
            // Zero means quarantined: excluded outright, not pushed down the list where a
            // long-enough top-k would find it again.
            if weight <= 0.0 {
                return None;
            }
            Some((
                hit.score * weight,
                hit.score,
                node.content.as_str(),
                node.source_type.clone(),
            ))
        })
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    ranked.truncate(3);
    for (score, cosine, content, source) in &ranked {
        println!("  {score:.3} (cos {cosine:.3})  [{source:?}]  {content}");
    }
    println!("  -- the self-description is gone: quarantined, not down-ranked.");

    // Following the edges reaches the fact the question does not resemble.
    println!("\nAssociative recall, 1 hop — reached through the graph, not the embedding:");
    for id in nexus.associative_recall(&index, &query, 2, 1) {
        let node = &nexus.get_node()[&id];
        let by_vector = index
            .search(&query, 8)
            .iter()
            .any(|hit| hit.node_id == id && hit.score >= RELEVANCE_FLOOR);
        let how = if by_vector { "vector" } else { "edge   " };
        println!("  [{how}]  {}", node.content);
    }

    // ── 4. Build the prompt ──────────────────────────────────────────────────────────────
    //
    // This is the whole point of the gate: what the model is allowed to treat as evidence.
    // Everything below came from a person or a document. Nothing the model said about
    // itself can appear here, no matter how well it matched the question.

    println!("\n── the block you hand to the LLM ────────────────────────────────────");
    println!("Answer using only the facts below. If they do not cover the question, say so.\n");
    for (_, _, content, source) in &ranked {
        println!("- ({source:?}) {content}");
    }
    println!("\nQuestion: {question}");
    println!("────────────────────────────────────────────────────────────────────");
    println!(
        "\nNote what is still in that block: the model-generated \"about 10 minutes\", ranked last\n\
         and contradicting the canon fact. Provenance excludes only what the model said about\n\
         ITSELF; other model output is down-weighted, not removed. That is the real boundary, and\n\
         the benchmarks report it the same way — self-echo 0/600, general model text 1/600."
    );

    // ── 5. What survives when memory runs out ────────────────────────────────────────────
    //
    // On a device this is not hypothetical. The protocol above is the least-read fact in the
    // store, which is exactly why recency and frequency policies drop it first.

    let before = nexus.len();
    let evicted = nexus.enforce_capacity(2);
    let protocol_survived = nexus.get_node().contains_key(&protocol);
    println!(
        "\nSqueezed {before} facts down to a budget of 2: evicted {evicted}, kept {}.",
        nexus.len()
    );
    println!(
        "The canon protocol survived: {protocol_survived} \
         (canon is exempt, so the budget is a floor rather than a ceiling)."
    );
}
