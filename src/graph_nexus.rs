//! Core Memory Graph Nexus
//! Storage for atomic memory nodes, edges, hyperedges, and cluster scope retrieval.
//! Not internally synchronized — wrap in a lock for concurrent use (the C-ABI handle does).

use crate::affect_overlay::AffectState;
use crate::canon::CanonLevel;
use crate::provenance::SourceType;
use crate::topology_24cell::Topology24Cell;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cmp::Ordering;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: String,
    pub content: String,
    pub summary: String,
    pub tags: Vec<String>,
    pub importance: f32,
    pub confidence: f32,
    pub source_type: SourceType,
    pub cluster_id: String,
    pub cell_id: u8,
    pub canon_level: CanonLevel,
    pub bridge_to_all: bool,
    pub is_ephemeral: bool,
    pub affect: AffectState,
    pub created_at: u64,
    pub last_accessed_at: u64,
    pub access_count: u32,
    /// Times a HUMAN confirmed this node was the right thing to recall. Never written by
    /// the engine's own behaviour: a node the model happened to retrieve is not evidence
    /// that retrieving it was correct, and treating it as such is self-echo with extra steps.
    #[serde(default)]
    pub confirmations: u32,
    /// Times a human said the recall was wrong or irrelevant.
    #[serde(default)]
    pub rejections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation_type: String, // "extractedTogether", "sameConversation", "relatedMeaning", "contradicts"
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperEdge {
    pub id: String,
    pub name: String,
    pub node_ids: Vec<String>,
    pub is_external: bool,
    pub cluster_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraphNexus {
    nodes: BTreeMap<String, MemoryNode>,
    edges: Vec<MemoryEdge>,
    hyperedges: Vec<HyperEdge>,
    topology: Topology24Cell,
    clock: u64,
    id_seed: u64,
}

impl MemoryGraphNexus {
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            edges: Vec::new(),
            hyperedges: Vec::new(),
            topology: Topology24Cell::new(),
            clock: 0,
            id_seed: 0x1234_5678_9ABC_DEF0,
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    fn next_id(&mut self) -> String {
        // SplitMix64 -> 16-hex-char id. Deterministic, no OS entropy.
        self.id_seed = self.id_seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.id_seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        alloc::format!("{:016x}", z)
    }

    /// Infer proper cluster routing based on node tags and provenance
    pub fn infer_cluster_for_node(&self, source_type: &SourceType, tags: &[String], explicit_cluster: Option<&str>) -> String {
        if let Some(cluster) = explicit_cluster {
            return cluster.to_string();
        }

        if source_type.is_self_description() {
            return "system".to_string();
        }

        let tag_set: BTreeSet<String> = tags.iter().map(|t| t.to_lowercase()).collect();

        if tag_set.contains("private") || tag_set.contains("confidential") || tag_set.contains("secret") {
            "private".to_string()
        } else if tag_set.contains("personal") || tag_set.contains("family") || tag_set.contains("home") {
            "personal".to_string()
        } else if tag_set.contains("work") || tag_set.contains("deadline") || tag_set.contains("task") {
            "work".to_string()
        } else if tag_set.contains("architecture") || tag_set.contains("code") || tag_set.contains("bug") {
            "system".to_string()
        } else if tag_set.contains("project") || tag_set.contains("goal") || tag_set.contains("milestone") {
            "project".to_string()
        } else {
            "shared".to_string()
        }
    }

    /// Add a new node to the graph nexus
    pub fn create_node(
        &mut self,
        content: String,
        summary: String,
        tags: Vec<String>,
        source_type: SourceType,
        cell_id: u8,
        explicit_cluster: Option<&str>,
        canon_level: CanonLevel,
    ) -> String {
        let id = self.next_id();
        let cluster_id = self.infer_cluster_for_node(&source_type, &tags, explicit_cluster);
        let now = self.next_tick();
        let cap = source_type.max_importance_cap();

        let node = MemoryNode {
            id: id.clone(),
            content,
            summary,
            tags,
            importance: 0.45f32.min(cap),
            confidence: 0.80,
            source_type,
            cluster_id,
            cell_id,
            canon_level,
            bridge_to_all: false,
            is_ephemeral: !canon_level.is_immortal(),
            affect: AffectState::default(),
            created_at: now,
            last_accessed_at: now,
            access_count: 1,
            confirmations: 0,
            rejections: 0,
        };

        self.nodes.insert(id.clone(), node);
        id
    }

    /// Create an undirected association between two nodes. No-op if either node is
    /// missing or the pair is already linked (dedup by unordered {source,target}).
    pub fn add_edge(&mut self, source_id: &str, target_id: &str, relation: &str, weight: f32) {
        if source_id == target_id
            || !self.nodes.contains_key(source_id)
            || !self.nodes.contains_key(target_id)
        {
            return;
        }
        let exists = self.edges.iter().any(|e| {
            (e.source_id == source_id && e.target_id == target_id)
                || (e.source_id == target_id && e.target_id == source_id)
        });
        if exists {
            return;
        }
        let edge_id = self.next_id();
        self.edges.push(MemoryEdge {
            id: edge_id,
            source_id: source_id.to_string(),
            target_id: target_id.to_string(),
            relation_type: relation.to_string(),
            weight,
        });
    }

    /// Auto-link node pairs whose embeddings are at least `threshold` cosine-similar,
    /// using the provided vector index. Edge weight = the cosine score. Returns the
    /// number of edges created. This is what turns isolated nodes into a graph.
    pub fn link_semantic(&mut self, index: &crate::vector_index::SimpleVectorIndex, threshold: f32) -> usize {
        let entries = index.entries(); // &[(node_id, embedding)]
        let mut to_add: Vec<(String, String, f32)> = Vec::new();
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let s = crate::vector_index::cosine_similarity(&entries[i].1, &entries[j].1);
                if s >= threshold {
                    to_add.push((entries[i].0.clone(), entries[j].0.clone(), s));
                }
            }
        }
        let before = self.edges.len();
        for (a, b, w) in to_add {
            self.add_edge(&a, &b, "relatedMeaning", w);
        }
        self.edges.len() - before
    }

    /// Single-table LSH linking (convenience: `n_tables = 1`).
    pub fn link_semantic_lsh(
        &mut self,
        index: &crate::vector_index::SimpleVectorIndex,
        threshold: f32,
        n_planes: usize,
    ) -> usize {
        self.link_semantic_lsh_multi(index, threshold, n_planes, 1)
    }

    /// Approximate semantic linking via MULTI-TABLE LSH — an O(N·b) alternative to the O(N²)
    /// all-pairs `link_semantic`. Each of `n_tables` independent hash tables projects vectors onto
    /// `n_planes` random hyperplanes (sign bits → bucket); a pair is a candidate if it shares a
    /// bucket in ANY table. More tables → higher recall at ~linear cost. Still APPROXIMATE and
    /// NOT HNSW — pairs missed by every table are missed. Returns edges created.
    pub fn link_semantic_lsh_multi(
        &mut self,
        index: &crate::vector_index::SimpleVectorIndex,
        threshold: f32,
        n_planes: usize,
        n_tables: usize,
    ) -> usize {
        let entries = index.entries();
        if entries.is_empty() {
            return 0;
        }
        let dim = entries[0].1.len();
        // Deterministic random hyperplanes (SplitMix64) — reproducible, no external crate.
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let mut rnd = || {
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };

        // Candidate pairs unioned across all tables (dedup by unordered index pair).
        let mut candidates: BTreeSet<(usize, usize)> = BTreeSet::new();
        for _ in 0..n_tables.max(1) {
            let planes: Vec<Vec<f32>> = (0..n_planes.min(63))
                .map(|_| (0..dim).map(|_| rnd()).collect())
                .collect();
            let mut buckets: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
            for (i, (_, v)) in entries.iter().enumerate() {
                let mut h = 0u64;
                for (b, p) in planes.iter().enumerate() {
                    let dot: f32 = v.iter().zip(p).map(|(a, b)| a * b).sum();
                    if dot >= 0.0 {
                        h |= 1 << b;
                    }
                }
                buckets.entry(h).or_default().push(i);
            }
            for ids in buckets.values() {
                for a in 0..ids.len() {
                    for b in (a + 1)..ids.len() {
                        candidates.insert((ids[a], ids[b]));
                    }
                }
            }
        }

        let mut to_add: Vec<(String, String, f32)> = Vec::new();
        for (i, j) in candidates {
            let s = crate::vector_index::cosine_similarity(&entries[i].1, &entries[j].1);
            if s >= threshold {
                to_add.push((entries[i].0.clone(), entries[j].0.clone(), s));
            }
        }
        let before = self.edges.len();
        for (a, b, w) in to_add {
            self.add_edge(&a, &b, "relatedMeaning", w);
        }
        self.edges.len() - before
    }

    /// Link nodes that share a RARE named entity — a sturdier alternative to cosine
    /// linking, which collapses into a clique on topically-dense corpora. Significant
    /// tokens are capitalized words (proper nouns: "Meridian", "Byte", "Tesla"). An
    /// entity appearing in more than `max_entity_freq` nodes is treated as non-discriminative
    /// (e.g. the subject "Alex" in every fact) and ignored — this is the IDF intuition.
    /// An edge is created when two nodes share at least `min_shared` rare entities;
    /// weight = the count of shared rare entities. Returns the number of edges created.
    pub fn link_by_entities(&mut self, max_entity_freq: usize, min_shared: usize) -> usize {
        fn entities(s: &str) -> BTreeSet<String> {
            s.split_whitespace()
                // Strip possessive first ("Alex's" -> "Alex"), then trim non-alphanumerics.
                .map(|w| w.split('\'').next().unwrap_or(w))
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|w| w.len() >= 3 && w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
                .collect()
        }
        let node_tokens: Vec<(String, BTreeSet<String>)> = self
            .nodes
            .values()
            .map(|n| (n.id.clone(), entities(&n.content)))
            .collect();

        let mut freq: BTreeMap<String, usize> = BTreeMap::new();
        for (_, toks) in &node_tokens {
            for t in toks {
                *freq.entry(t.clone()).or_insert(0) += 1;
            }
        }

        let mut to_add: Vec<(String, String, f32)> = Vec::new();
        for i in 0..node_tokens.len() {
            for j in (i + 1)..node_tokens.len() {
                let shared = node_tokens[i]
                    .1
                    .intersection(&node_tokens[j].1)
                    .filter(|t| freq[*t] <= max_entity_freq)
                    .count();
                if shared >= min_shared {
                    to_add.push((node_tokens[i].0.clone(), node_tokens[j].0.clone(), shared as f32));
                }
            }
        }
        let before = self.edges.len();
        for (a, b, w) in to_add {
            self.add_edge(&a, &b, "sharedEntity", w);
        }
        self.edges.len() - before
    }

    /// Prune redundant edges by Forman-Ricci curvature. F(e) = 4 - deg(u) - deg(v) + 3·triangles.
    /// HIGH F means the edge sits inside a dense cluster with many alternative paths — redundant,
    /// safe to drop (noise reduction). NEGATIVE F means a bridge/bottleneck between clusters —
    /// structurally load-bearing, KEPT. (Note: the old doc-claim "prune F<0.10" was backwards —
    /// that would cut the bridges.) An edge is removed only if its F exceeds `min_curvature`
    /// AND neither endpoint would be left isolated. Returns the number of edges pruned.
    pub fn prune_redundant_edges(&mut self, min_curvature: f32) -> usize {
        use crate::ricci_curvature::FormanRicciCalculator as F;
        // degree + adjacency (sets) for triangle counting.
        let mut deg: BTreeMap<&str, usize> = BTreeMap::new();
        let mut adj: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for e in &self.edges {
            *deg.entry(e.source_id.as_str()).or_insert(0) += 1;
            *deg.entry(e.target_id.as_str()).or_insert(0) += 1;
            adj.entry(e.source_id.as_str()).or_default().insert(e.target_id.as_str());
            adj.entry(e.target_id.as_str()).or_default().insert(e.source_id.as_str());
        }
        // Score each edge; collect redundant candidates (highest F first — drop the most redundant).
        let mut scored: Vec<(usize, f32)> = self
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let (u, v) = (e.source_id.as_str(), e.target_id.as_str());
                let tri = match (adj.get(u), adj.get(v)) {
                    (Some(nu), Some(nv)) => nu.intersection(nv).count(),
                    _ => 0,
                };
                (i, F::calculate_edge_curvature(deg[u], deg[v], tri))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Live degree, decremented as we prune, so we never isolate a node.
        let mut live_deg = deg.clone();
        let mut remove: BTreeSet<usize> = BTreeSet::new();
        for (i, f) in scored {
            if f <= min_curvature {
                break; // rest are bridges/weak — keep
            }
            let e = &self.edges[i];
            let (u, v) = (e.source_id.as_str(), e.target_id.as_str());
            if live_deg[u] > 1 && live_deg[v] > 1 {
                remove.insert(i);
                *live_deg.get_mut(u).unwrap() -= 1;
                *live_deg.get_mut(v).unwrap() -= 1;
            }
        }
        let before = self.edges.len();
        let mut idx = 0;
        self.edges.retain(|_| {
            let keep = !remove.contains(&idx);
            idx += 1;
            keep
        });
        before - self.edges.len()
    }

    /// Directly-linked neighbor node ids of `node_id` (undirected).
    pub fn neighbors(&self, node_id: &str) -> Vec<String> {
        let mut out = Vec::new();
        for e in &self.edges {
            if e.source_id == node_id {
                out.push(e.target_id.clone());
            } else if e.target_id == node_id {
                out.push(e.source_id.clone());
            }
        }
        out
    }

    /// Breadth-first closure over edges: all node ids reachable from `seeds` within
    /// `hops` steps (seeds included). This is the associative-recall expansion —
    /// it surfaces linked facts a flat similarity search would miss.
    pub fn expand(&self, seeds: &[String], hops: usize) -> BTreeSet<String> {
        // Build adjacency once (O(E)) instead of scanning the edge list per visited node.
        let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for e in &self.edges {
            adj.entry(e.source_id.as_str()).or_default().push(e.target_id.as_str());
            adj.entry(e.target_id.as_str()).or_default().push(e.source_id.as_str());
        }
        let mut seen: BTreeSet<String> =
            seeds.iter().filter(|s| self.nodes.contains_key(*s)).cloned().collect();
        let mut frontier: Vec<String> = seen.iter().cloned().collect();
        for _ in 0..hops {
            let mut next = Vec::new();
            for id in &frontier {
                for &nb in adj.get(id.as_str()).map(|v| v.as_slice()).unwrap_or(&[]) {
                    if seen.insert(nb.to_string()) {
                        next.push(nb.to_string());
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        seen
    }

    /// Associative recall: seed-SET entry + graph expansion. Takes the top `k_seeds`
    /// cosine matches as entry points (not just top-1 — a single entry is fragile if the
    /// true seed isn't rank-1), then expands `hops` steps over edges. Returns node ids
    /// ranked as: the seeds in cosine order first, then expanded neighbors. This surfaces
    /// linked facts a flat similarity search misses (benchmark: 100% vs 0% reach on a
    /// sparse semantic graph, up from 60% with a single top-1 entry point).
    pub fn associative_recall(
        &self,
        index: &crate::vector_index::SimpleVectorIndex,
        query: &[f32],
        k_seeds: usize,
        hops: usize,
    ) -> Vec<String> {
        let seeds: Vec<String> = index
            .search(query, k_seeds)
            .into_iter()
            .map(|r| r.node_id)
            .collect();
        let reached = self.expand(&seeds, hops);
        // Seeds first (ranked), then the rest of the reached set.
        let mut out = Vec::with_capacity(reached.len());
        for s in &seeds {
            if reached.contains(s) {
                out.push(s.clone());
            }
        }
        for id in &reached {
            if !seeds.contains(id) {
                out.push(id.clone());
            }
        }
        out
    }

    /// Evict under capacity pressure until at most `max_nodes` remain. Canon nodes
    /// (`is_ephemeral == false`, i.e. L1/L2) are NEVER evicted — that is the whole point
    /// of canon: important facts survive memory pressure while noise is dropped. Among
    /// ephemeral nodes, evicts the least valuable first: lowest importance, then oldest
    /// last-access (LRU), then lowest access_count. Edges touching removed nodes are pruned.
    ///
    /// Importance leads deliberately. The engine's whole claim is that value is neither
    /// recent nor frequent, and ordering by recency first contradicted it: the logical clock
    /// gives every node a distinct last-access tick, so importance never broke a tie and human
    /// feedback would have been inert. With no feedback recorded, all nodes of one provenance
    /// share the same importance and the order falls back to recency exactly as before.
    /// If canon alone exceeds `max_nodes`, all canon is kept (capacity yields to immortality).
    /// Returns the number of nodes evicted.
    pub fn enforce_capacity(&mut self, max_nodes: usize) -> usize {
        if self.nodes.len() <= max_nodes {
            return 0;
        }
        let mut evictable: Vec<(String, u64, u32, f32)> = self
            .nodes
            .values()
            .filter(|n| n.is_ephemeral) // canon is exempt
            .map(|n| (n.id.clone(), n.last_accessed_at, n.access_count, n.importance))
            .collect();
        // Least valuable first: lowest importance, then oldest access, then fewest accesses.
        evictable.sort_by(|a, b| {
            a.3.partial_cmp(&b.3)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });
        let want_remove = self.nodes.len() - max_nodes;
        let remove_n = want_remove.min(evictable.len());
        let mut removed = 0;
        for (id, ..) in evictable.iter().take(remove_n) {
            self.nodes.remove(id);
            self.edges.retain(|e| e.source_id != *id && e.target_id != *id);
            removed += 1;
        }
        removed
    }

    /// Set the emotional disposition of a stored node — the write path that feeds
    /// `AffectOverlayEngine::compute_mood` at recall time. Returns false if the node
    /// does not exist.
    pub fn set_node_affect(&mut self, node_id: &str, affect: AffectState) -> bool {
        match self.nodes.get_mut(node_id) {
            Some(n) => {
                n.affect = affect;
                true
            }
            None => false,
        }
    }

    /// Number of non-ephemeral (canon) nodes currently stored.
    pub fn canon_count(&self) -> usize {
        self.nodes.values().filter(|n| !n.is_ephemeral).count()
    }

    /// Record that a node was accessed (recalled): bump its access counter and touch its
    /// last-access time. Frequency/recency-based eviction reads these — which is exactly why
    /// a rarely-queried but critical fact needs the explicit canon flag to survive: usage
    /// signals do not capture importance.
    pub fn record_access(&mut self, node_id: &str) {
        let now = self.next_tick();
        if let Some(n) = self.nodes.get_mut(node_id) {
            n.access_count = n.access_count.saturating_add(1);
            n.last_accessed_at = now;
        }
    }

    /// Record a HUMAN verdict on a recall: was this node the right thing to surface?
    ///
    /// This is the only reward signal the engine accepts, and the restriction is the point.
    /// A signal derived from the model's own behaviour ("it retrieved this, so it must have
    /// been useful") closes a loop onto the model's output — the same self-echo the provenance
    /// guard exists to break, only harder to see. So the caller must invoke this from a human
    /// confirmation, never from its own agent loop.
    ///
    /// The verdict moves RETENTION, not relevance. `importance` is read by capacity eviction
    /// and by nothing in the ranking path, so a confirmed fact survives pressure longer while
    /// the order of search results is untouched — a measured lesson from this codebase, where
    /// folding value multipliers into the rank cost 86% -> 29% recall@1.
    ///
    /// Two invariants hold regardless of how much feedback arrives:
    /// - **The provenance ceiling is absolute.** Importance is clamped to
    ///   `source_type.max_importance_cap()`, so applause cannot promote model self-description
    ///   to the standing of a user's own words.
    /// - **Rejection outweighs confirmation** (-0.20 vs +0.15): a recall that was wrong costs
    ///   the user more than a right one gains, so the engine is quicker to doubt than to trust.
    ///
    /// Returns false if the node does not exist.
    pub fn record_human_feedback(&mut self, node_id: &str, helpful: bool) -> bool {
        const CONFIRM_STEP: f32 = 0.15;
        const REJECT_STEP: f32 = 0.20;

        match self.nodes.get_mut(node_id) {
            Some(n) => {
                let cap = n.source_type.max_importance_cap();
                if helpful {
                    n.confirmations = n.confirmations.saturating_add(1);
                    n.importance = (n.importance + CONFIRM_STEP).min(cap);
                } else {
                    n.rejections = n.rejections.saturating_add(1);
                    n.importance = (n.importance - REJECT_STEP).max(0.0);
                }
                true
            }
            None => false,
        }
    }

    /// Promote nodes a human has confirmed at least `min_confirmations` times to canon, making
    /// them eviction-proof. Deliberately an explicit sweep rather than an automatic side effect
    /// of feedback: turning a fact immortal is a decision, and the caller should be the one
    /// making it at a moment of their choosing.
    ///
    /// Self-description is never promoted no matter how many confirmations it collects — the
    /// provenance quarantine is not a ranking preference that enough votes can overturn.
    /// Nodes already canon are left alone. Returns how many were promoted.
    pub fn promote_confirmed_to_canon(&mut self, min_confirmations: u32, level: CanonLevel) -> usize {
        if !level.is_canon() {
            return 0;
        }
        let mut promoted = 0;
        for n in self.nodes.values_mut() {
            if n.canon_level.is_canon()
                || n.source_type.is_self_description()
                || n.confirmations < min_confirmations
            {
                continue;
            }
            n.canon_level = level;
            n.is_ephemeral = false;
            promoted += 1;
        }
        promoted
    }

    pub fn edges(&self) -> &[MemoryEdge] {
        &self.edges
    }

    pub fn hyperedges(&self) -> &[HyperEdge] {
        &self.hyperedges
    }

    pub fn topology(&self) -> &Topology24Cell {
        &self.topology
    }

    pub fn get_node(&self) -> &BTreeMap<String, MemoryNode> {
        &self.nodes
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Retrieve visible cluster scope for a given persona persona_name
    pub fn get_scope_clusters(&self, persona_name: &str, include_system: bool, include_private: bool) -> BTreeSet<String> {
        let mut scope = BTreeSet::new();
        
        // Base public clusters
        scope.insert("shared".to_string());
        scope.insert("work".to_string());
        scope.insert("personal".to_string());
        scope.insert("project".to_string());

        // Persona specific cluster
        scope.insert(persona_name.to_string());

        if include_system {
            scope.insert("system".to_string());
        }
        if include_private {
            scope.insert("private".to_string());
        }

        scope
    }
}

impl Default for MemoryGraphNexus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation_and_cluster_routing() {
        let mut nexus = MemoryGraphNexus::new();

        // Self description should go to system
        let id_sys = nexus.create_node(
            "I am a hypergraph bot".to_string(),
            "self desc".to_string(),
            vec!["architecture".to_string()],
            SourceType::LlmSelfDescription,
            11,
            None,
            CanonLevel::None,
        );

        let sys_node = nexus.nodes.get(&id_sys).unwrap();
        assert_eq!(sys_node.cluster_id, "system");

        // User utterance should go to shared
        let id_user = nexus.create_node(
            "User prefers concise answers".to_string(),
            "user preference".to_string(),
            vec!["user_profile".to_string()],
            SourceType::UserUtterance,
            2,
            None,
            CanonLevel::L2Foundational,
        );

        let user_node = nexus.nodes.get(&id_user).unwrap();
        assert_eq!(user_node.cluster_id, "shared");
        assert_eq!(user_node.canon_level, CanonLevel::L2Foundational);
    }

    #[test]
    fn test_edges_and_traversal() {
        let mut nexus = MemoryGraphNexus::new();
        let mk = |n: &mut MemoryGraphNexus, t: &str| {
            n.create_node(t.into(), t.into(), vec![], SourceType::UserUtterance, 2, None, CanonLevel::None)
        };
        let a = mk(&mut nexus, "A");
        let b = mk(&mut nexus, "B");
        let c = mk(&mut nexus, "C");

        // Chain A—B—C. C is NOT directly linked to A.
        nexus.add_edge(&a, &b, "relatedMeaning", 0.9);
        nexus.add_edge(&b, &c, "relatedMeaning", 0.9);
        // Dedup + missing-node guards.
        nexus.add_edge(&a, &b, "relatedMeaning", 0.9); // duplicate ignored
        nexus.add_edge(&a, "nonexistent", "x", 1.0);   // missing target ignored
        assert_eq!(nexus.edges().len(), 2);

        // Neighbors are undirected.
        assert_eq!(nexus.neighbors(&b).len(), 2);
        assert_eq!(nexus.neighbors(&a), vec![b.clone()]);

        // 1 hop from A reaches B but not C; 2 hops reaches C (the multi-hop payoff).
        let one = nexus.expand(&[a.clone()], 1);
        assert!(one.contains(&b) && !one.contains(&c));
        let two = nexus.expand(&[a.clone()], 2);
        assert!(two.contains(&c));
    }

    #[test]
    fn test_lsh_linking_finds_near_duplicates() {
        use crate::vector_index::SimpleVectorIndex;
        let mut g = MemoryGraphNexus::new();
        let mut idx = SimpleVectorIndex::new(4);
        let mk = |g: &mut MemoryGraphNexus| {
            g.create_node("x".into(), String::new(), vec![], SourceType::UserUtterance, 2, None, CanonLevel::None)
        };
        let a = mk(&mut g);
        let b = mk(&mut g);
        let c = mk(&mut g);
        idx.insert(a.clone(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert(b.clone(), vec![0.98, 0.02, 0.0, 0.0]).unwrap(); // near-duplicate of a
        idx.insert(c.clone(), vec![0.0, 0.0, 1.0, 0.0]).unwrap();   // orthogonal

        g.link_semantic_lsh(&idx, 0.9, 12);
        assert!(g.neighbors(&a).contains(&b), "near-duplicate should be linked");
        assert!(!g.neighbors(&a).contains(&c), "orthogonal vector should not link");
    }

    #[test]
    fn test_ricci_prune_keeps_bridges_drops_redundant() {
        let mut g = MemoryGraphNexus::new();
        let mk = |g: &mut MemoryGraphNexus, t: &str| {
            g.create_node(t.into(), t.into(), vec![], SourceType::UserUtterance, 2, None, CanonLevel::None)
        };
        // Two triangles A-B-C and D-E-F, joined by a single bridge C-D.
        let (a, b, c) = (mk(&mut g, "A"), mk(&mut g, "B"), mk(&mut g, "C"));
        let (d, e, f) = (mk(&mut g, "D"), mk(&mut g, "E"), mk(&mut g, "F"));
        for (x, y) in [(&a, &b), (&b, &c), (&c, &a), (&d, &e), (&e, &f), (&f, &d), (&c, &d)] {
            g.add_edge(x, y, "relatedMeaning", 1.0);
        }
        assert_eq!(g.edges().len(), 7);

        // Prune redundant (high-F) triangle edges; the bridge C-D has negative F → kept.
        let pruned = g.prune_redundant_edges(2.0);
        assert!(pruned >= 1, "at least one redundant triangle edge should be pruned");
        // The bridge survives (it is the only path between the two clusters).
        assert!(g.neighbors(&c).contains(&d), "bridge C-D must be kept");
        // No node was isolated.
        for id in [&a, &b, &c, &d, &e, &f] {
            assert!(!g.neighbors(id).is_empty(), "no node should be isolated by pruning");
        }
    }

    #[test]
    fn test_entity_linking_ignores_frequent_subject() {
        let mut nexus = MemoryGraphNexus::new();
        let mk = |n: &mut MemoryGraphNexus, t: &str| {
            n.create_node(t.into(), t.into(), vec![], SourceType::UserUtterance, 2, None, CanonLevel::None)
        };
        // "Alex" appears in 3 nodes (frequent → ignored). "Meridian" (freq 2) links the pair.
        let a = mk(&mut nexus, "Alex works at Meridian");
        let b = mk(&mut nexus, "Meridian builds robots");
        let c = mk(&mut nexus, "Alex likes tea");
        let _d = mk(&mut nexus, "Alex plays guitar");

        // max_entity_freq = 2: "Alex" (freq 3) is ignored, "Meridian" (freq 2) links a–b.
        let edges = nexus.link_by_entities(2, 1);
        assert_eq!(edges, 1, "only the Meridian pair should link");
        assert!(nexus.neighbors(&a).contains(&b));
        assert!(!nexus.neighbors(&a).contains(&c), "Alex is too frequent to link a–c");
    }

    #[test]
    fn test_associative_recall_seed_set() {
        use crate::vector_index::SimpleVectorIndex;
        let mut nexus = MemoryGraphNexus::new();
        let mk = |n: &mut MemoryGraphNexus, t: &str| {
            n.create_node(t.into(), t.into(), vec![], SourceType::UserUtterance, 2, None, CanonLevel::None)
        };
        let seed = mk(&mut nexus, "Alex works at Meridian");
        let target = mk(&mut nexus, "Meridian builds robots");
        let noise = mk(&mut nexus, "unrelated fact");

        // Seed and target are linked; target is embedding-distant from the query vector.
        nexus.add_edge(&seed, &target, "relatedMeaning", 0.9);

        let mut idx = SimpleVectorIndex::new(2);
        idx.insert(seed.clone(), vec![1.0, 0.0]).unwrap();   // close to query
        idx.insert(target.clone(), vec![0.0, 1.0]).unwrap(); // far from query
        idx.insert(noise.clone(), vec![-1.0, 0.0]).unwrap();

        let query = [1.0f32, 0.0];
        // Flat top-1 would only return the seed; associative recall reaches the target via the edge.
        let recalled = nexus.associative_recall(&idx, &query, 2, 1);
        assert!(recalled.contains(&seed));
        assert!(recalled.contains(&target), "seed-set recall must reach the linked target");
    }

    #[test]
    fn test_canon_survives_capacity_pressure() {
        let mut nexus = MemoryGraphNexus::new();
        // 3 canon facts (immortal) created first, then never touched again (oldest = LRU targets).
        let mut canon_ids = Vec::new();
        for i in 0..3 {
            canon_ids.push(nexus.create_node(
                alloc::format!("canon fact {i}"), String::new(), vec![],
                SourceType::UserUtterance, 2, None, CanonLevel::L2Foundational,
            ));
        }
        // 20 ephemeral noise nodes.
        for i in 0..20 {
            nexus.create_node(
                alloc::format!("noise {i}"), String::new(), vec![],
                SourceType::LlmGeneration, 5, None, CanonLevel::None,
            );
        }
        assert_eq!(nexus.len(), 23);
        assert_eq!(nexus.canon_count(), 3);

        // Squeeze hard: keep only 5 nodes. A plain LRU would evict the old canon facts.
        let evicted = nexus.enforce_capacity(5);
        assert_eq!(evicted, 18); // 23 -> 5
        assert_eq!(nexus.len(), 5);
        // All canon survived despite being the oldest nodes.
        assert_eq!(nexus.canon_count(), 3);
        for id in &canon_ids {
            assert!(nexus.get_node().contains_key(id), "canon fact was wrongly evicted");
        }
    }

    #[test]
    fn test_human_feedback_respects_the_provenance_ceiling() {
        let mut nexus = MemoryGraphNexus::new();
        let gen = nexus.create_node(
            "model said something plausible".to_string(), String::new(), vec![],
            SourceType::LlmGeneration, 5, None, CanonLevel::None,
        );
        let selfd = nexus.create_node(
            "i am an assistant with a memory".to_string(), String::new(), vec![],
            SourceType::LlmSelfDescription, 11, None, CanonLevel::None,
        );

        // Applause cannot lift a node past what its provenance allows: 0.70 for generation,
        // 0.40 for self-description, no matter how many times a human confirms it.
        for _ in 0..20 {
            assert!(nexus.record_human_feedback(&gen, true));
            assert!(nexus.record_human_feedback(&selfd, true));
        }
        assert_eq!(nexus.get_node()[&gen].importance, 0.70);
        assert_eq!(nexus.get_node()[&selfd].importance, 0.40);
        assert_eq!(nexus.get_node()[&gen].confirmations, 20);

        // Rejection bottoms out at zero rather than going negative.
        for _ in 0..20 {
            assert!(nexus.record_human_feedback(&gen, false));
        }
        assert_eq!(nexus.get_node()[&gen].importance, 0.0);
        assert_eq!(nexus.get_node()[&gen].rejections, 20);

        assert!(!nexus.record_human_feedback("no-such-node", true));
    }

    #[test]
    fn test_feedback_moves_retention_not_ranking() {
        use crate::vector_index::SimpleVectorIndex;

        let mut nexus = MemoryGraphNexus::new();
        let mut idx = SimpleVectorIndex::new(3);
        let mut ids = Vec::new();
        for i in 0..3 {
            let id = nexus.create_node(
                alloc::format!("fact {i}"), String::new(), vec![],
                SourceType::UserUtterance, 2, None, CanonLevel::None,
            );
            idx.insert(id.clone(), vec![1.0 - i as f32 * 0.1, i as f32 * 0.1, 0.0]).unwrap();
            ids.push(id);
        }

        let query = [1.0f32, 0.0, 0.0];
        let before: Vec<String> = idx.search(&query, 3).into_iter().map(|r| r.node_id).collect();

        // Bury the top hit under rejections and lift the last one with confirmations.
        for _ in 0..5 {
            nexus.record_human_feedback(&ids[0], false);
            nexus.record_human_feedback(&ids[2], true);
        }

        // RANKING: identical. Feedback is not allowed anywhere near relevance.
        let after: Vec<String> = idx.search(&query, 3).into_iter().map(|r| r.node_id).collect();
        assert_eq!(before, after, "human feedback must not reorder search results");

        // RETENTION: under pressure the rejected fact goes first and the confirmed one stays,
        // even though the rejected one is the closest match to the query.
        nexus.enforce_capacity(1);
        assert!(!nexus.get_node().contains_key(&ids[0]), "rejected node should be evicted first");
        assert!(nexus.get_node().contains_key(&ids[2]), "confirmed node should survive");
    }

    #[test]
    fn test_promotion_to_canon_is_explicit_and_quarantine_holds() {
        let mut nexus = MemoryGraphNexus::new();
        let fact = nexus.create_node(
            "the user is allergic to penicillin".to_string(), String::new(), vec![],
            SourceType::UserUtterance, 2, None, CanonLevel::None,
        );
        let selfd = nexus.create_node(
            "i keep notes about the user".to_string(), String::new(), vec![],
            SourceType::LlmSelfDescription, 11, None, CanonLevel::None,
        );
        for _ in 0..3 {
            nexus.record_human_feedback(&fact, true);
            nexus.record_human_feedback(&selfd, true);
        }

        // Confirmations alone change nothing — promotion is a separate, deliberate act.
        assert_eq!(nexus.canon_count(), 0);

        assert_eq!(nexus.promote_confirmed_to_canon(3, CanonLevel::L1Project), 1);
        assert_eq!(nexus.canon_count(), 1);
        assert!(nexus.get_node()[&fact].canon_level.is_canon());
        // Enough votes never buy self-description its way out of quarantine.
        assert!(!nexus.get_node()[&selfd].canon_level.is_canon());

        // Idempotent: an already-canon node is not promoted twice.
        assert_eq!(nexus.promote_confirmed_to_canon(3, CanonLevel::L1Project), 0);
        // And the promoted fact now survives what would otherwise evict it.
        nexus.enforce_capacity(0);
        assert!(nexus.get_node().contains_key(&fact));
    }
}
