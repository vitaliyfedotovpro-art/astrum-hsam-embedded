//! Recall@k benchmark: f32 vs int8-quantized embedding storage on the SAME graph.
//!
//! Ground truth = exact f32 cosine top-k. We then store vectors as int8 (per-vector
//! symmetric scale) and measure how many of the true f32 neighbors int8 still returns.
//! Recall is a numeric property (identical on host and MCU), so we run on host.
//!
//! Two metrics:
//!   (A) Fidelity recall@k  = |int8_topk ∩ f32_exact_topk| / k   (does quantization change neighbors?)
//!   (B) Task precision@k   = fraction of returned items in the query's cluster (real-task impact)
//!
//! Data: C clusters of unit vectors + Gaussian noise (fixed-seed, reproducible).
//! Run: cargo run --release --example recall_int8

use astrum_memory::cosine_similarity;

// ---- deterministic RNG (SplitMix64) — no external crate ----
struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32 // [0,1)
    }
    fn gauss(&mut self) -> f32 {
        // Box–Muller
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }
    fn rand_unit(&mut self, d: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| self.gauss()).collect();
        normalize(&mut v);
        v
    }
}

fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Per-vector symmetric int8 quantization. Returns (int8 codes, scale).
/// Storage cost per vector = D bytes + 4 (scale) vs 4*D for f32.
fn quantize_i8(v: &[f32]) -> (Vec<i8>, f32) {
    let m = v.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let scale = if m > 0.0 { m / 127.0 } else { 1.0 };
    let q = v
        .iter()
        .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

fn dequantize(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|&c| c as f32 * scale).collect()
}

/// Indices of the top-k corpus vectors by cosine to `query`.
fn topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_similarity(query, v)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

const D: usize = 384;
const C: usize = 50; // clusters
const PER: usize = 40; // vectors per cluster -> N = 2000
const Q: usize = 300; // queries

/// Run one regime at a given noise level. Returns
/// (fidelity@5, f32_precision@5, int8_precision@5) in percent.
fn run(sigma: f32) -> (f64, f64, f64) {
    let k = 5usize;
    let mut rng = Rng(0xDEAD_BEEF_1234_5678);

    let centers: Vec<Vec<f32>> = (0..C).map(|_| rng.rand_unit(D)).collect();

    let mut corpus_f32: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    for (c, center) in centers.iter().enumerate() {
        for _ in 0..PER {
            let mut v: Vec<f32> = (0..D).map(|i| center[i] + sigma * rng.gauss()).collect();
            normalize(&mut v);
            corpus_f32.push(v);
            labels.push(c);
        }
    }

    // int8 store -> dequantized corpus (models int8 storage + f32 query = asymmetric)
    let corpus_i8: Vec<Vec<f32>> = corpus_f32
        .iter()
        .map(|v| {
            let (q, s) = quantize_i8(v);
            dequantize(&q, s)
        })
        .collect();

    let mut queries: Vec<(Vec<f32>, usize)> = Vec::new();
    for _ in 0..Q {
        let c = (rng.u64() as usize) % C;
        let mut v: Vec<f32> = (0..D)
            .map(|i| centers[c][i] + sigma * rng.gauss())
            .collect();
        normalize(&mut v);
        queries.push((v, c));
    }

    let (mut fidelity, mut pf32, mut pi8) = (0.0f64, 0.0f64, 0.0f64);
    for (qv, qc) in &queries {
        let true_top = topk(qv, &corpus_f32, k);
        let i8_top = topk(qv, &corpus_i8, k);
        let true_k: std::collections::HashSet<usize> = true_top.iter().copied().collect();
        fidelity += i8_top.iter().filter(|i| true_k.contains(i)).count() as f64 / k as f64;
        pf32 += true_top.iter().filter(|&&i| labels[i] == *qc).count() as f64 / k as f64;
        pi8 += i8_top.iter().filter(|&&i| labels[i] == *qc).count() as f64 / k as f64;
    }
    let n = Q as f64;
    (100.0 * fidelity / n, 100.0 * pf32 / n, 100.0 * pi8 / n)
}

fn main() {
    println!(
        "=== int8 vs f32 recall@5 — N={} D={} clusters={} queries={} ===",
        C * PER,
        D,
        C,
        Q
    );
    println!(
        "mem/vector: f32={} B  int8={} B  ({:.2}x smaller)",
        4 * D,
        D + 4,
        (4 * D) as f32 / (D + 4) as f32
    );
    println!();
    println!("  sigma | task | fidelity  f32-prec@5  int8-prec@5   delta");
    println!("  ------+------+-------------------------------------------");
    for &sigma in &[0.03f32, 0.06, 0.10, 0.15, 0.20] {
        let (fid, pf, pi) = run(sigma);
        let regime = if pf > 95.0 {
            "easy"
        } else if pf > 60.0 {
            "med "
        } else {
            "hard"
        };
        println!(
            "  {:.2}  | {} | {:5.1}%     {:5.1}%      {:5.1}%     {:+.2} pp",
            sigma,
            regime,
            fid,
            pf,
            pi,
            pi - pf
        );
    }
    println!();
    println!("fidelity = int8 top-5 that match exact-f32 top-5.");
    println!("prec@5   = returned items in the query's true cluster (the real task).");
    println!("delta    = int8 task precision minus f32 task precision (negative = int8 worse).");
}
