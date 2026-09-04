//! Query latency: how long a recall actually takes, and how it scales.
//!
//! Footprint told us what a fact COSTS to store; this tells us what a query costs to answer.
//! The engine's search is an exact linear scan, so the honest expectation is time linear in
//! N x D. This measures whether that holds and what the constant is.
//!
//! SCOPE, stated up front: these are DESKTOP numbers (this machine's CPU, wall clock). They
//! are NOT a Cortex-M4 latency claim and must never be quoted as one — an MCU at ~100 MHz
//! without SIMD is one to two orders of magnitude slower, and only a real board can say by how
//! much. What DOES transfer is the shape (linear in N, linear in D) and the operation count
//! per query, which is fixed by the algorithm: N x D multiply-accumulates for f32, the same
//! count in i8-to-f32 for the quantized index.
//!
//! Run: cargo run --release --example latency

use astrum_memory::{Int8VectorIndex, SimpleVectorIndex};
use std::time::Instant;

// Deterministic RNG (SplitMix64) — same generator as the other benches, no external crate.
struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self, d: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..d)
            .map(|_| (self.u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }
}

const D: usize = 384;
const TOP_K: usize = 5;
const QUERIES: usize = 200;

fn main() {
    println!("# Query latency — DESKTOP wall clock, NOT an MCU claim");
    println!("D={D}, top_k={TOP_K}, {QUERIES} queries per point, exact linear scan\n");
    println!("|     N | f32 us/query | int8 us/query | f32 ns/node | int8 ns/node | MACs/query |");
    println!("|------:|-------------:|--------------:|------------:|-------------:|-----------:|");

    for &n in &[128usize, 512, 2000, 5000, 10000] {
        let mut rng = Rng(0x5EED_1234);
        let mut f32_idx = SimpleVectorIndex::new(D);
        let mut i8_idx = Int8VectorIndex::new(D);
        for i in 0..n {
            let v = rng.unit(D);
            f32_idx.insert(format!("n{i}"), v.clone()).unwrap();
            i8_idx.insert(format!("n{i}"), v).unwrap();
        }
        let queries: Vec<Vec<f32>> = (0..QUERIES).map(|_| rng.unit(D)).collect();

        // Warm up caches/branch predictors so the first point is not penalised.
        for q in queries.iter().take(10) {
            std::hint::black_box(f32_idx.search(q, TOP_K));
            std::hint::black_box(i8_idx.search(q, TOP_K));
        }

        let t0 = Instant::now();
        for q in &queries {
            std::hint::black_box(f32_idx.search(q, TOP_K));
        }
        let f32_us = t0.elapsed().as_secs_f64() * 1e6 / QUERIES as f64;

        let t0 = Instant::now();
        for q in &queries {
            std::hint::black_box(i8_idx.search(q, TOP_K));
        }
        let i8_us = t0.elapsed().as_secs_f64() * 1e6 / QUERIES as f64;

        println!(
            "| {:>5} | {:>12.1} | {:>13.1} | {:>11.1} | {:>12.1} | {:>10} |",
            n,
            f32_us,
            i8_us,
            f32_us * 1000.0 / n as f64,
            i8_us * 1000.0 / n as f64,
            n * D
        );
    }

    println!("\nns/node should stay flat if the scan is linear; MACs/query = N x D is fixed by");
    println!("the algorithm and is what an MCU budget must be computed from. A device figure");
    println!("requires measurement on real silicon — do not extrapolate these numbers to one.");
}
