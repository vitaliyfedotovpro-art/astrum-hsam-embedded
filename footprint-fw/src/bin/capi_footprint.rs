#![no_std]
#![no_main]

//! Footprint of the SHIPPABLE path: every node goes in through the C-ABI
//! (`astrum_memory_add_node`), not through the Rust API. This is what a partner
//! linking `libastrum_memory.a` actually pays per fact, including the C-side
//! overhead the Rust benchmark does not see (node-id `CString` handed back over
//! the boundary, JSON result strings).
//!
//! Built with `--features capi-int8`, so the C-ABI is backed by `Int8VectorIndex`
//! — the configuration behind the headline 801 B/node. Run:
//!   cd footprint-fw && cargo run --release --bin capi_footprint

extern crate alloc;

use alloc::ffi::CString;
use alloc::vec::Vec;

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use embedded_alloc::LlffHeap as Heap;

use astrum_memory::c_api::{
    astrum_memory_add_node, astrum_memory_create, astrum_memory_destroy,
    astrum_memory_free_string, astrum_memory_node_count, astrum_memory_search,
};

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    let _ = hprintln!("PANIC");
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}

const HEAP_SIZE: usize = 3 * 1024 * 1024; // 3 MiB heap in SSRAM (board has 4 MiB total)
static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];

/// Embedding dimension. 384 = typical small sentence-embedder (MiniLM).
const DIM: usize = 384;

/// Deterministic pseudo-embedding (LCG), same generator as the Rust-path benchmark
/// so the two numbers are comparable.
fn embedding(i: usize) -> Vec<f32> {
    let mut v: Vec<f32> = Vec::with_capacity(DIM);
    let mut s = i as u32 + 1;
    for _ in 0..DIM {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(((s >> 8) as f32) / 65_536.0 - 128.0);
    }
    v
}

/// Insert `n` nodes through the C-ABI and return heap bytes used at peak
/// (engine still alive), plus the node count the engine reports back.
fn build_and_measure_capi(n: usize) -> (usize, usize) {
    let handle = astrum_memory_create();

    for i in 0..n {
        // Realistic short fact string, handed over as a C string like a real caller would.
        let content = CString::new(alloc::format!("fact number {} about entity {}", i, i % 97))
            .unwrap();
        let v = embedding(i);
        let id = astrum_memory_add_node(
            handle,
            content.as_ptr(),
            core::ptr::null(),
            0, // source_type = user_utterance
            2, // cell_id
            0, // canon_level = none
            v.as_ptr(),
            DIM,
        );
        // A caller that leaks this is a caller with a bug; free it as the API demands.
        astrum_memory_free_string(id);
    }

    // Exercise the recall path so nothing is optimized away, and free its result.
    let q = embedding(0);
    let json = astrum_memory_search(handle, q.as_ptr(), DIM, 2, 5);
    astrum_memory_free_string(json);

    let used = HEAP.used(); // measured while the engine is still alive
    let count = astrum_memory_node_count(handle);
    astrum_memory_destroy(handle);
    (used, count)
}

#[entry]
fn main() -> ! {
    unsafe {
        HEAP.init(core::ptr::addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE);
    }
    let base = HEAP.used();
    let _ = hprintln!("=== Astrum HSAM C-ABI footprint on Cortex-M4 (thumbv7em) ===");
    let _ = hprintln!("DIM={}, index backend = {}", DIM, astrum_memory::CAPI_INDEX_KIND);
    let _ = hprintln!("   N |  heap bytes | B/node | nodes");

    for &n in &[128usize, 256, 512, 1000] {
        let (used, count) = build_and_measure_capi(n);
        let used = used - base;
        let _ = hprintln!("{:>4} | {:>11} | {:>6} | {}", n, used, used / n, count);
    }

    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}
