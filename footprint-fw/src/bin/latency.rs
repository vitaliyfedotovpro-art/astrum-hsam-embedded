#![no_std]
#![no_main]

//! Query cost on Cortex-M4 — SCALING, not seconds.
//!
//! READ THIS BEFORE QUOTING ANY NUMBER FROM HERE. QEMU is not cycle-accurate: it does not
//! model the M4 pipeline, flash wait states, or the FPU's real timing, and SysTick advances
//! against virtual time. So the absolute figures below are NOT device latency and must never
//! be presented as such. **A latency claim requires a real board.**
//!
//! What this run does establish, and what QEMU is adequate for:
//!   - the scan is linear in N (cost per node stays flat as N grows),
//!   - the arithmetic actually executes on the target ISA without surprises,
//!   - the per-query operation count, which is fixed by the algorithm: N x D
//!     multiply-accumulates. That number is exact and is what an MCU time budget
//!     should be computed from once the per-MAC cost of the real part is known.
//!
//! Run: cd footprint-fw && cargo run --release --bin latency

extern crate alloc;

use alloc::format;
use alloc::vec::Vec;

use cortex_m::peripheral::syst::SystClkSource;
use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use embedded_alloc::LlffHeap as Heap;

use astrum_memory::{Int8VectorIndex, SimpleVectorIndex};

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    let _ = hprintln!("PANIC");
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}

const HEAP_SIZE: usize = 3 * 1024 * 1024;
static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];

const DIM: usize = 384;
const TOP_K: usize = 5;
const QUERIES: usize = 20;

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

    // SysTick as a free-running down-counter over the widest reload it supports.
    let mut syst = unsafe { cortex_m::Peripherals::steal() }.SYST;
    syst.set_clock_source(SystClkSource::Core);
    syst.set_reload(0x00FF_FFFF);
    syst.clear_current();
    syst.enable_counter();

    let _ = hprintln!("=== Query cost on Cortex-M4 (QEMU) — SCALING ONLY, not device latency ===");
    let _ = hprintln!("DIM={}, top_k={}, {} queries per point", DIM, TOP_K, QUERIES);
    let _ = hprintln!("    N | f32 ticks/query | /node | int8 ticks/query | /node |   MACs/query");

    for &n in &[128usize, 256, 512, 1000] {
        let mut f32_idx = SimpleVectorIndex::new(DIM);
        let mut i8_idx = Int8VectorIndex::new(DIM);
        for i in 0..n {
            let v = embedding(i);
            let _ = f32_idx.insert(format!("n{}", i), v.clone());
            let _ = i8_idx.insert(format!("n{}", i), v);
        }
        let q = embedding(7);

        // SysTick counts DOWN; a wrap would corrupt the delta, so each query is timed
        // separately and the ticks are summed.
        let mut f32_ticks: u32 = 0;
        for _ in 0..QUERIES {
            let start = cortex_m::peripheral::SYST::get_current();
            let hits = f32_idx.search(&q, TOP_K);
            let end = cortex_m::peripheral::SYST::get_current();
            core::hint::black_box(&hits);
            f32_ticks = f32_ticks.wrapping_add(start.wrapping_sub(end) & 0x00FF_FFFF);
        }

        let mut i8_ticks: u32 = 0;
        for _ in 0..QUERIES {
            let start = cortex_m::peripheral::SYST::get_current();
            let hits = i8_idx.search(&q, TOP_K);
            let end = cortex_m::peripheral::SYST::get_current();
            core::hint::black_box(&hits);
            i8_ticks = i8_ticks.wrapping_add(start.wrapping_sub(end) & 0x00FF_FFFF);
        }

        let f32_per = f32_ticks / QUERIES as u32;
        let i8_per = i8_ticks / QUERIES as u32;
        let _ = hprintln!(
            "{:>5} | {:>15} | {:>5} | {:>16} | {:>5} | {:>12}",
            n,
            f32_per,
            f32_per / n as u32,
            i8_per,
            i8_per / n as u32,
            n * DIM
        );
    }

    let _ = hprintln!("ticks/node flat across N => the scan is linear, as the algorithm says.");
    let _ = hprintln!("These ticks are QEMU virtual time. Device latency needs real silicon.");

    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}
