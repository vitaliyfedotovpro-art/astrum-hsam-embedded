#![no_std]
#![no_main]

//! Self-echo contamination@3 on Cortex-M4 (QEMU), int8 index, 5000 nodes.
//! Ports the quality section of the original desktop bench_scale to bare metal.
//! 200 user "truths", each with 2 high-cosine LLM self-echo near-copies; 100 canon;
//! filler LLM-generation up to 5000. For each truth we query and check the top-3:
//!   flat  = plain cosine top-3, count LLM (self-description | generation) items;
//!   guard = relevance floor (>=0.25) + EXCLUDE quarantined (rank_multiplier==0) + rerank.
//! Expectation: guard ~0% while flat leaks.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use embedded_alloc::LlffHeap as Heap;

use astrum_memory::{CanonLevel, Int8VectorIndex, MemoryGraphNexus, SourceType};

#[global_allocator]
static HEAP: Heap = Heap::empty();
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    let _ = hprintln!("PANIC");
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}
const HEAP_SIZE: usize = 3 * 1024 * 1024; // fits mps2-an386 4MB RAM
static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];

const DIM: usize = 384;

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn vec(&mut self) -> Vec<f32> {
        let v: Vec<f32> = (0..DIM).map(|_| self.unit()).collect();
        let n = libm::sqrtf(v.iter().map(|x| x * x).sum::<f32>()).max(1e-9);
        v.iter().map(|x| x / n).collect()
    }
    fn near(&mut self, base: &[f32], noise: f32) -> Vec<f32> {
        let v: Vec<f32> = base.iter().map(|b| b + self.unit() * noise).collect();
        let n = libm::sqrtf(v.iter().map(|x| x * x).sum::<f32>()).max(1e-9);
        v.iter().map(|x| x / n).collect()
    }
}

#[entry]
fn main() -> ! {
    unsafe {
        HEAP.init(core::ptr::addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE);
    }
    let n = 3000usize; // sized to the board's 3 MiB heap; effect matches 5k
    let mut rng = Rng(0xC0FFEE);
    let mut g = MemoryGraphNexus::new();
    let mut idx = Int8VectorIndex::new(DIM);

    // 200 truths + 2 self-echo each.
    let mut truths: Vec<Vec<f32>> = Vec::new();
    for _ in 0..200 {
        let tv = rng.vec();
        let id = g.create_node("truth".into(), String::new(), Vec::new(), SourceType::UserUtterance, 2, None, CanonLevel::None);
        idx.insert(id, tv.clone()).unwrap();
        for _ in 0..2 {
            let ev = rng.near(&tv, 0.15);
            let eid = g.create_node("echo".into(), String::new(), Vec::new(), SourceType::LlmSelfDescription, 11, None, CanonLevel::None);
            idx.insert(eid, ev).unwrap();
        }
        truths.push(tv);
    }
    // 100 canon.
    for _ in 0..100 {
        let id = g.create_node("canon".into(), String::new(), Vec::new(), SourceType::UserUtterance, 2, None, CanonLevel::L2Foundational);
        idx.insert(id, rng.vec()).unwrap();
    }
    // filler LLM-generation up to N.
    while g.len() < n {
        let id = g.create_node("filler".into(), String::new(), Vec::new(), SourceType::LlmGeneration, 5, None, CanonLevel::None);
        idx.insert(id.clone(), rng.vec()).unwrap();
        g.record_access(&id);
    }

    let _ = hprintln!("=== contamination@3 on Cortex-M4 (int8 idx, N={}) ===", g.len());

    // Track self-echo (LlmSelfDescription) and general-LLM (LlmGeneration) leaks separately.
    let (mut flat_echo, mut flat_gen) = (0usize, 0usize);
    let (mut guard_echo, mut guard_gen) = (0usize, 0usize);
    const FLOOR: f32 = 0.25;
    let nodes = g.get_node();
    let classify = |st: &SourceType| -> (bool, bool) {
        (
            matches!(st, SourceType::LlmSelfDescription),
            matches!(st, SourceType::LlmGeneration),
        )
    };
    for tv in truths.iter() {
        let cand = idx.search(tv, 12);
        // flat: top-3 by cosine.
        for r in cand.iter().take(3) {
            let (e, gn) = classify(&nodes.get(&r.node_id).unwrap().source_type);
            flat_echo += e as usize;
            flat_gen += gn as usize;
        }
        // guard: floor + exclude quarantined (rank_multiplier==0, i.e. self-description) + rerank.
        let mut w: Vec<(f32, &SourceType)> = cand
            .iter()
            .filter(|r| r.score >= FLOOR)
            .filter(|r| nodes.get(&r.node_id).unwrap().source_type.rank_multiplier() > 0.0)
            .map(|r| {
                let nd = nodes.get(&r.node_id).unwrap();
                (r.score * nd.source_type.rank_multiplier(), &nd.source_type)
            })
            .collect();
        w.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        for (_, st) in w.iter().take(3) {
            let (e, gn) = classify(st);
            guard_echo += e as usize;
            guard_gen += gn as usize;
        }
    }

    let denom = 200 * 3;
    let pct = |x: usize| (x * 1000 / denom) as f32 / 10.0; // one decimal
    let _ = hprintln!("denom = {} (top-3 slots over 200 truths)", denom);
    let _ = hprintln!(
        "SELF-ECHO (LlmSelfDescription):  flat {}/{}={}%   guard {}/{}={}%",
        flat_echo, denom, pct(flat_echo), guard_echo, denom, pct(guard_echo)
    );
    let _ = hprintln!(
        "general-LLM (LlmGeneration):     flat {}/{}={}%   guard {}/{}={}%",
        flat_gen, denom, pct(flat_gen), guard_gen, denom, pct(guard_gen)
    );
    let _ = hprintln!(
        "ALL LLM (echo+gen):              flat {}%   guard {}%",
        pct(flat_echo + flat_gen), pct(guard_echo + guard_gen)
    );
    let _ = hprintln!("heap_used_bytes={}", HEAP.used());

    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}
