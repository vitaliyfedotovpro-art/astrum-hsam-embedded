#![no_std]
#![no_main]

//! What happens when the heap runs out.
//!
//! An embedded consumer needs the real answer to this before anything else: a memory engine
//! that hangs the device when the heap fills is not shippable, and "it probably panics" is not
//! an answer. So this firmware deliberately runs a small heap into the ground and reports what
//! the engine does on the way there.
//!
//! Deliberately a tiny heap (64 KiB ≈ 40 nodes at 384-dim f32) so exhaustion arrives quickly.
//!
//! Run: cd footprint-fw && cargo run --release --bin oom

extern crate alloc;

use alloc::format;
use alloc::vec::Vec;

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use embedded_alloc::LlffHeap as Heap;

use astrum_memory::{CanonLevel, MemoryGraphNexus, SimpleVectorIndex, SourceType};

#[global_allocator]
static HEAP: Heap = Heap::empty();

/// The engine's own panic path on this board. Reached via `handle_alloc_error` when an
/// allocation fails, which is what makes exhaustion observable at all.
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    let _ = hprintln!("[oom] PANIC — allocation failure reached the panic handler");
    let _ = hprintln!("[oom] on a real device the `runtime` feature's handler spins here forever");
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}

const HEAP_SIZE: usize = 64 * 1024;
static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];

const DIM: usize = 384;

fn embedding(i: usize) -> Vec<f32> {
    let mut v: Vec<f32> = Vec::with_capacity(DIM);
    let mut s = i as u32 + 1;
    for _ in 0..DIM {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(((s >> 8) as f32) / 65_536.0 - 128.0);
    }
    v
}

#[entry]
fn main() -> ! {
    unsafe {
        HEAP.init(core::ptr::addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE);
    }
    let _ = hprintln!("=== Heap exhaustion behaviour (64 KiB heap, DIM={}) ===", DIM);

    // The recommended pattern: keep the engine inside a budget you chose, rather than
    // discovering the ceiling by hitting it. Cap derived from the measured bytes/node.
    const BUDGET_NODES: usize = 25;

    let base = HEAP.used();
    {
    let mut nexus = MemoryGraphNexus::new();
    let mut index = SimpleVectorIndex::new(DIM);

    for i in 0..200usize {
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
        let _ = index.insert(id, embedding(i));

        // Bounded operation: evict down to the budget and shed the evicted embeddings.
        if nexus.len() > BUDGET_NODES {
            let evicted = nexus.enforce_capacity(BUDGET_NODES);
            let nodes = nexus.get_node();
            let dropped = index.retain_ids(|id| nodes.contains_key(id));
            if i % 50 == 0 {
                let _ = hprintln!(
                    "  i={:>3} nodes={} vectors={} heap={} (evicted {} / dropped {})",
                    i,
                    nexus.len(),
                    index.len(),
                    HEAP.used(),
                    evicted,
                    dropped
                );
            }
        }
    }

    let _ = hprintln!(
        "[oom] SURVIVED 200 inserts inside a {}-node budget: nodes={} vectors={} heap={}/{}",
        BUDGET_NODES,
        nexus.len(),
        index.len(),
        HEAP.used(),
        HEAP_SIZE
    );
    let _ = hprintln!("[oom] steady state — bounded engine does not grow into the ceiling");
    }

    // Everything above is dropped here. If teardown is clean the heap returns to where it
    // started; anything left behind is a leak, and this is the cheapest place to see one.
    let _ = hprintln!("[oom] after dropping the engine: heap={} (started at {})", HEAP.used(), base);

    // Now the unbounded case, on purpose: keep inserting with no budget until the heap dies.
    let _ = hprintln!("[oom] now running UNBOUNDED until the allocator fails...");
    let mut nexus2 = MemoryGraphNexus::new();
    let mut index2 = SimpleVectorIndex::new(DIM);
    for i in 0..10_000usize {
        let content = format!("unbounded fact {}", i);
        let id = nexus2.create_node(
            content.clone(),
            content,
            Vec::new(),
            SourceType::UserUtterance,
            2,
            None,
            CanonLevel::None,
        );
        let _ = index2.insert(id, embedding(i));
        if i % 10 == 0 {
            let _ = hprintln!("  unbounded i={} nodes={} heap={}", i, nexus2.len(), HEAP.used());
        }
    }

    let _ = hprintln!("[oom] UNREACHABLE if exhaustion behaves as expected");
    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}
