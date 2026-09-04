//! In-memory linear-scan cosine vector index.
//! Exact top-K similarity search over a flat Vec — O(N) per query, no external
//! dependencies. Adequate up to tens of thousands of vectors; swap in an ANN
//! index behind the same interface beyond that.

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }

    (dot / (libm::sqrtf(norm_a) * libm::sqrtf(norm_b))).clamp(-1.0, 1.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub node_id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleVectorIndex {
    dimension: usize,
    vectors: Vec<(String, Vec<f32>)>,
}

impl SimpleVectorIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: Vec::new(),
        }
    }

    pub fn insert(&mut self, node_id: String, embedding: Vec<f32>) -> Result<(), String> {
        if embedding.len() != self.dimension {
            return Err(alloc::format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len()
            ));
        }
        self.vectors.push((node_id, embedding));
        Ok(())
    }

    /// Exact top-k by cosine. Keeps a bounded (score, position) shortlist instead of scoring
    /// everything into a Vec and sorting it: the old form cloned one heap `String` per stored
    /// vector on EVERY query and then sorted all N, so a 3,000-node engine did 3,000
    /// allocations per recall — the kind of churn that fragments a bare-metal heap. Only the
    /// k winners' ids are cloned now.
    ///
    /// Measured, so the claim stays honest: this bought ~17% on desktop (247 -> 200 ns/node),
    /// NOT the large win the allocation count suggests. The scan is compute-bound, not
    /// allocation-bound. The reason to keep it is the heap behaviour on a device, where
    /// allocation churn costs more than the arithmetic saved.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<VectorSearchResult> {
        if query.len() != self.dimension || self.vectors.is_empty() || top_k == 0 {
            return Vec::new();
        }

        // Shortlist kept sorted by score descending; ties keep insertion order, matching the
        // stable sort this replaced.
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(top_k);
        for (i, (_, vec)) in self.vectors.iter().enumerate() {
            let score = cosine_similarity(query, vec);
            let pos = best.partition_point(|&(s, _)| s >= score);
            if pos < top_k {
                best.insert(pos, (score, i));
                if best.len() > top_k {
                    best.pop();
                }
            }
        }

        best.into_iter()
            .map(|(score, i)| VectorSearchResult {
                node_id: self.vectors[i].0.clone(),
                score,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Read-only view of (node_id, embedding) pairs — used for building semantic edges.
    pub fn entries(&self) -> &[(String, Vec<f32>)] {
        &self.vectors
    }

    /// Drop every vector whose id fails `keep`. Eviction must reach the index, not just the
    /// graph: the embedding is the bulk of a node's cost, so dropping the node alone frees
    /// almost nothing. Returns the number of vectors removed.
    pub fn retain_ids<F: Fn(&str) -> bool>(&mut self, keep: F) -> usize {
        let before = self.vectors.len();
        self.vectors.retain(|(id, _)| keep(id));
        before - self.vectors.len()
    }
}

/// int8-quantized cosine index. Stores each embedding as i8 codes + a single f32
/// scale (per-vector symmetric quantization), so memory is `dimension + 4` bytes
/// per vector versus `4 * dimension` for `SimpleVectorIndex` — ~4x smaller at
/// typical dims. In cosine the scale cancels, so search needs no dequantization
/// and allocates nothing per comparison. Recall loss vs f32 is <=0.3 pp on task
/// (see examples/recall_int8.rs). A sibling type — the f32 index stays the exact
/// reference; opt into int8 where SRAM matters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Int8VectorIndex {
    dimension: usize,
    // (node_id, i8 codes, scale)
    vectors: Vec<(String, Vec<i8>, f32)>,
}

impl Int8VectorIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: Vec::new(),
        }
    }

    /// Quantize and store. Symmetric per-vector: scale = max|x| / 127.
    pub fn insert(&mut self, node_id: String, embedding: Vec<f32>) -> Result<(), String> {
        if embedding.len() != self.dimension {
            return Err(alloc::format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len()
            ));
        }
        let m = embedding.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let scale = if m > 0.0 { m / 127.0 } else { 1.0 };
        let codes: Vec<i8> = embedding
            .iter()
            .map(|&x| libm::roundf(x / scale).clamp(-127.0, 127.0) as i8)
            .collect();
        self.vectors.push((node_id, codes, scale));
        Ok(())
    }

    /// Cosine top-k. The per-vector scale cancels in cosine, so we compute directly from i8
    /// codes against the f32 query — no dequant, no alloc. Selection is the same bounded
    /// shortlist as `SimpleVectorIndex::search`, for the same reason: no per-candidate
    /// allocation, no full sort.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<VectorSearchResult> {
        if query.len() != self.dimension || self.vectors.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let mut norm_q = 0.0f32;
        for &x in query {
            norm_q += x * x;
        }
        norm_q = libm::sqrtf(norm_q);
        if norm_q <= 0.0 {
            return Vec::new();
        }

        let mut best: Vec<(f32, usize)> = Vec::with_capacity(top_k);
        for (i, (_, codes, _scale)) in self.vectors.iter().enumerate() {
            let mut dot = 0.0f32;
            let mut norm_c = 0.0f32;
            for j in 0..codes.len() {
                let c = codes[j] as f32;
                dot += c * query[j];
                norm_c += c * c;
            }
            norm_c = libm::sqrtf(norm_c);
            let score = if norm_c <= 0.0 {
                0.0
            } else {
                (dot / (norm_q * norm_c)).clamp(-1.0, 1.0)
            };

            let pos = best.partition_point(|&(s, _)| s >= score);
            if pos < top_k {
                best.insert(pos, (score, i));
                if best.len() > top_k {
                    best.pop();
                }
            }
        }

        best.into_iter()
            .map(|(score, i)| VectorSearchResult {
                node_id: self.vectors[i].0.clone(),
                score,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Drop every vector whose id fails `keep` — see `SimpleVectorIndex::retain_ids`.
    pub fn retain_ids<F: Fn(&str) -> bool>(&mut self, keep: F) -> usize {
        let before = self.vectors.len();
        self.vectors.retain(|(id, _, _)| keep(id));
        before - self.vectors.len()
    }
}

/// The index type the C-ABI (and therefore the shippable staticlib and its snapshots)
/// is built against. Default is the exact f32 index — 1,949 B/node measured on Cortex-M4.
/// Building with `--features capi-int8` swaps in `Int8VectorIndex` — 801 B/node measured,
/// at a task cost of <=0.3 pp (both numbers in RESULTS.md). The two builds write
/// INCOMPATIBLE snapshots; `Snapshot::index_kind` names which one produced a file.
#[cfg(not(feature = "capi-int8"))]
pub type CapiIndex = SimpleVectorIndex;
#[cfg(feature = "capi-int8")]
pub type CapiIndex = Int8VectorIndex;

/// Name of the active C-ABI index backend, as stored in a snapshot. `"f32"` or `"int8"`.
pub const CAPI_INDEX_KIND: &str = if cfg!(feature = "capi-int8") { "int8" } else { "f32" };

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0];

        assert_eq!(cosine_similarity(&v1, &v2), 1.0);
        assert_eq!(cosine_similarity(&v1, &v3), 0.0);
    }

    #[test]
    fn test_vector_index_search() {
        let mut index = SimpleVectorIndex::new(3);
        index.insert("node_a".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        index.insert("node_b".to_string(), vec![0.8, 0.2, 0.0]).unwrap();
        index.insert("node_c".to_string(), vec![0.0, 1.0, 0.0]).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let hits = index.search(&query, 2);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].node_id, "node_a");
        assert_eq!(hits[0].score, 1.0);
        assert_eq!(hits[1].node_id, "node_b");
    }

    #[test]
    fn test_int8_index_preserves_ranking() {
        let mut idx = Int8VectorIndex::new(3);
        idx.insert("a".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert("b".to_string(), vec![0.8, 0.2, 0.0]).unwrap();
        idx.insert("c".to_string(), vec![0.0, 1.0, 0.0]).unwrap();

        let hits = idx.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(hits.len(), 3);
        // Same ranking as the f32 index; [1,0,0] round-trips exactly -> score 1.0.
        assert_eq!(hits[0].node_id, "a");
        assert!((hits[0].score - 1.0).abs() < 1e-4);
        assert_eq!(hits[1].node_id, "b");
        assert_eq!(hits[2].node_id, "c");
        // Dimension guard.
        assert!(idx.insert("bad".to_string(), vec![1.0, 0.0]).is_err());
    }
}
