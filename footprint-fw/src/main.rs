#![no_std]
#![no_main]

//! Footprint firmware: measures the REAL heap high-water-mark of Astrum HSAM
//! on a Cortex-M4 (thumbv7em) under QEMU (mps2-an386). We insert N nodes, each
//! with a DIM-dim f32 embedding, and read the allocator's `used()` at peak
//! (structures still alive). The slope of used-vs-N is the honest bytes/node
//! on a 32-bit MCU — the number that replaces the fabricated "1.4 MB SRAM".

extern crate alloc;

use alloc::format;
use alloc::vec::Vec;

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use embedded_alloc::LlffHeap as Heap;

use astrum_memory::{CanonLevel, Int8VectorIndex, MemoryGraphNexus, SimpleVectorIndex, SourceType};

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    let _ = hprintln!("PANIC");
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}

const HEAP_SIZE: usize = 3 * 1024 * 1024; // 3 MiB heap in SSRAM
static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];

/// Embedding dimension. 384 = typical small sentence-embedder (MiniLM), f32.
const DIM: usize = 384;

/// Build a graph of `n` nodes with DIM-dim f32 embeddings; return heap bytes
/// used at peak (while the structures are still alive).
fn build_and_measure(n: usize) -> usize {
    let mut nexus = MemoryGraphNexus::new();
    let mut index = SimpleVectorIndex::new(DIM);

    for i in 0..n {
        // Realistic short fact string (heap-allocated content + summary).
        let content = format!("fact number {} about entity {}", i, i % 97);
        let id = nexus.create_node(
            content.clone(),
            content,
            Vec::new(),
            SourceType::UserUtterance,
            2,
            None,
            CanonLevel::None,
        );

        // Deterministic pseudo-embedding (LCG) so every node carries a real vector.
        let mut v: Vec<f32> = Vec::with_capacity(DIM);
        let mut s = i as u32 + 1;
        for _ in 0..DIM {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push(((s >> 8) as f32) / 65_536.0 - 128.0);
        }
        let _ = index.insert(id, v);
    }

    // Exercise the recall path so nothing is optimized away.
    let q: Vec<f32> = (0..DIM).map(|k| k as f32).collect();
    let hits = index.search(&q, 5);
    core::hint::black_box(&hits);
    core::hint::black_box(&nexus);
    core::hint::black_box(&index);

    HEAP.used() // measured while nexus + index are still alive
}

/// Same graph, but embeddings stored in the int8 index (i8 codes + scale).
fn build_and_measure_i8(n: usize) -> usize {
    let mut nexus = MemoryGraphNexus::new();
    let mut index = Int8VectorIndex::new(DIM);

    for i in 0..n {
        let content = format!("fact number {} about entity {}", i, i % 97);
        let id = nexus.create_node(
            content.clone(),
            content,
            Vec::new(),
            SourceType::UserUtterance,
            2,
            None,
            CanonLevel::None,
        );

        let mut v: Vec<f32> = Vec::with_capacity(DIM);
        let mut s = i as u32 + 1;
        for _ in 0..DIM {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push(((s >> 8) as f32) / 65_536.0 - 128.0);
        }
        let _ = index.insert(id, v); // quantized to i8 inside
    }

    let q: Vec<f32> = (0..DIM).map(|k| k as f32).collect();
    let hits = index.search(&q, 5);
    core::hint::black_box(&hits);
    core::hint::black_box(&nexus);
    core::hint::black_box(&index);

    HEAP.used()
}

#[entry]
fn main() -> ! {
    unsafe {
        HEAP.init(core::ptr::addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE);
    }
    let base = HEAP.used();
    let _ = hprintln!("=== Astrum HSAM footprint on Cortex-M4 (thumbv7em) ===");
    let _ = hprintln!("DIM={} (f32 vs int8 vector index, same graph)", DIM);
    let _ = hprintln!("  N | f32 bytes  B/node | int8 bytes  B/node | saved");

    for &n in &[128usize, 256, 512, 1000] {
        let f32_used = build_and_measure(n) - base;
        let i8_used = build_and_measure_i8(n) - base;
        let _ = hprintln!(
            "{:>4} | {:>9} {:>6} | {:>9} {:>6} | {}%",
            n,
            f32_used,
            f32_used / n,
            i8_used,
            i8_used / n,
            100 - (i8_used * 100 / f32_used)
        );
    }

    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}
