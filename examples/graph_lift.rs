//! Graph-lift benchmark: does HSAM's associative recall beat FLAT cosine when the
//! target is embedding-distant from the query but linked by a co-occurrence edge?
//!
//! Uses the REAL engine: MemoryGraphNexus + SimpleVectorIndex + associative_recall.
//! Honest setup (can fail): every query has ONE seed (close to query) and ONE target
//! (far from query) joined by a co-occurrence edge. Plus a background of distractors
//! and NOISE edges (random links) so expansion also drags in junk. We report both
//! recall (did we get the target) and precision cost (how big/dirty the result set is).
//!
//! Run: cargo run --release --example graph_lift

use astrum_memory::{CanonLevel, MemoryGraphNexus, SimpleVectorIndex, SourceType};

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn uf(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let u1 = self.uf().max(1e-7);
        let u2 = self.uf();
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }
    fn unit(&mut self, d: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| self.gauss()).collect();
        norm(&mut v);
        v
    }
}
fn norm(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

const D: usize = 384;
const QN: usize = 300; // query cases (each adds a seed + a target)
const DISTRACT: usize = 1400; // background noise facts
const K: usize = 10; // top-k budget for flat search / seed set
const HOPS: usize = 1;

/// Run one regime with `avg_noise_deg` random edges per node. Returns
/// (flat_recall%, assoc_recall%, avg_assoc_setsize, assoc_precision%).
fn run(avg_noise_deg: usize) -> (f64, f64, f64, f64) {
    let mut rng = Rng(0xABCD_EF12_3456_789Bu64 ^ (avg_noise_deg as u64).wrapping_mul(0x9E3779B1));
    let mut nexus = MemoryGraphNexus::new();
    let mut index = SimpleVectorIndex::new(D);

    let mk = |nexus: &mut MemoryGraphNexus, tag: &str| {
        nexus.create_node(
            tag.into(),
            String::new(),
            Vec::new(),
            SourceType::UserUtterance,
            2,
            None,
            CanonLevel::None,
        )
    };

    // Background distractors.
    let mut all_ids: Vec<String> = Vec::new();
    for _ in 0..DISTRACT {
        let id = mk(&mut nexus, "bg");
        let v = rng.unit(D);
        index.insert(id.clone(), v).unwrap();
        all_ids.push(id);
    }

    // Query cases: seed (near query) + target (far from query) + co-occurrence edge.
    let mut cases: Vec<(Vec<f32>, String)> = Vec::new(); // (query, target_id)
    for _ in 0..QN {
        // seed direction; query = seed + small noise (query lands near the seed).
        let s = rng.unit(D);
        let mut q: Vec<f32> = s.iter().map(|&x| x + 0.05 * rng.gauss()).collect();
        norm(&mut q);
        let seed_id = mk(&mut nexus, "seed");
        index.insert(seed_id.clone(), s.clone()).unwrap();
        all_ids.push(seed_id.clone());

        // target = independent random direction => embedding-distant from q (cos ~ 0).
        let t = rng.unit(D);
        let target_id = mk(&mut nexus, "target");
        index.insert(target_id.clone(), t).unwrap();
        all_ids.push(target_id.clone());

        // co-occurrence edge seed<->target (the associative bridge).
        nexus.add_edge(&seed_id, &target_id, "coOccur", 1.0);

        cases.push((q, target_id));
    }

    // NOISE edges: random links so expansion also drags in junk (fair cost).
    let total_noise = avg_noise_deg * all_ids.len();
    for _ in 0..total_noise {
        let a = &all_ids[(rng.u64() as usize) % all_ids.len()];
        let b = &all_ids[(rng.u64() as usize) % all_ids.len()];
        nexus.add_edge(a, b, "noise", 0.1);
    }

    // Measure.
    let (mut flat_hit, mut assoc_hit, mut setsum, mut relsum) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (q, target) in &cases {
        // FLAT: top-K cosine.
        let flat: Vec<String> = index.search(q, K).into_iter().map(|r| r.node_id).collect();
        if flat.iter().any(|id| id == target) {
            flat_hit += 1.0;
        }
        // ASSOCIATIVE: K seeds, HOPS expansion over edges (real engine).
        let assoc = nexus.associative_recall(&index, q, K, HOPS);
        if assoc.iter().any(|id| id == target) {
            assoc_hit += 1.0;
        }
        setsum += assoc.len() as f64;
        // precision proxy: fraction of returned that are "relevant" (the seed or target
        // near THIS query). We approximate relevance by: returned id is a seed/target
        // whose embedding cos to q >= 0.3 OR is the linked target. Cheap proxy.
        let rel = assoc.iter().filter(|id| *id == target).count();
        relsum += rel as f64 / assoc.len().max(1) as f64;
    }
    let n = QN as f64;
    (
        100.0 * flat_hit / n,
        100.0 * assoc_hit / n,
        setsum / n,
        100.0 * relsum / n,
    )
}

fn main() {
    println!("=== graph lift — target is embedding-distant but co-occurrence-linked ===",);
    println!(
        "nodes={} (bg {} + {} seed/target pairs)  D={}  top-k={}  hops={}",
        DISTRACT + 2 * QN,
        DISTRACT,
        QN,
        D,
        K,
        HOPS
    );
    println!();
    println!("  noise-edges/node | flat recall | assoc recall | assoc set size | target-in-set");
    println!("  -----------------+-------------+--------------+----------------+--------------");
    for &deg in &[0usize, 1, 3, 8] {
        let (flat, assoc, size, prec) = run(deg);
        println!(
            "  {:>6}           |   {:5.1}%    |   {:5.1}%     |    {:6.1}      |   {:5.2}%",
            deg, flat, assoc, size, prec
        );
    }
    println!();
    println!("flat recall  = target found by plain cosine top-k (expected ~0: target is far).");
    println!("assoc recall = target found by associative_recall (seed -> edge -> target).");
    println!("set size     = avg #ids returned by associative recall (expansion cost).");
    println!(
        "target-in-set= share of the returned set that is the wanted target (precision proxy)."
    );
}
