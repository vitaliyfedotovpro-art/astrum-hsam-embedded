#![no_std]
#![no_main]

//! Canon retention under capacity pressure on Cortex-M4 (QEMU).
//! Ports the desktop bench_canon to bare metal: 5 critical canon facts (rarely
//! accessed, oldest) + 45 noise nodes (frequently accessed). Squeeze 50 -> 10.
//! HSAM pins canon; recency-LRU and frequency baselines drop it. Proves canon
//! retention is NOT a by-construction triviality (importance != recency/frequency).

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use embedded_alloc::LlffHeap as Heap;

use astrum_memory::{CanonLevel, MemoryGraphNexus, SourceType};

#[global_allocator]
static HEAP: Heap = Heap::empty();
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    let _ = hprintln!("PANIC");
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}
const HEAP_SIZE: usize = 1024 * 1024;
static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];

#[entry]
fn main() -> ! {
    unsafe {
        HEAP.init(core::ptr::addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE);
    }
    let _ = hprintln!("[canon] start");

    let mut nexus = MemoryGraphNexus::new();

    // 5 critical canon facts, created FIRST (oldest logical ticks).
    for f in [
        "CRITICAL: allergic to penicillin",
        "CRITICAL: blood type O-negative",
        "CRITICAL: emergency contact Dr Chen",
        "CRITICAL: has a pacemaker implant",
        "CRITICAL: avoid MRI, metal implant",
    ] {
        nexus.create_node(f.into(), String::new(), Vec::new(), SourceType::UserUtterance, 2, None, CanonLevel::L2Foundational);
    }
    // 45 noise nodes, created AFTER canon (newer ticks).
    for i in 0..45 {
        let mut c = String::from("routine sensor log ");
        c.push_str(itoa(i).as_str());
        nexus.create_node(c, String::new(), Vec::new(), SourceType::LlmGeneration, 5, None, CanonLevel::None);
    }
    // Access pattern: noise queried 3-8 times; canon never touched.
    let noise_ids: Vec<String> = nexus
        .get_node()
        .values()
        .filter(|n| n.is_ephemeral)
        .map(|n| n.id.clone())
        .collect();
    for (i, id) in noise_ids.iter().enumerate() {
        for _ in 0..((i % 6) + 3) {
            nexus.record_access(id);
        }
    }

    let _ = hprintln!("[canon] built {} nodes", nexus.len());
    // Baselines from snapshot BEFORE enforce_capacity.
    let (recency_canon, freq_canon) = {
        let all = nexus.get_node();
        let mut v: Vec<(bool, u64, u32)> = all.values().map(|n| (!n.is_ephemeral, n.last_accessed_at, n.access_count)).collect();
        // recency: newest last_accessed first
        v.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        let rc = v.iter().take(10).filter(|x| x.0).count();
        // frequency: most-accessed first
        v.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)));
        let fc = v.iter().take(10).filter(|x| x.0).count();
        (rc, fc)
    };

    // HSAM: enforce_capacity protects canon.
    nexus.enforce_capacity(10);
    let hsam_canon = nexus.canon_count();

    let _ = hprintln!("=== canon retention on Cortex-M4 (50 -> 10 nodes) ===");
    let _ = hprintln!("HSAM (canon pinned):        {}/5 = {}%", hsam_canon, 100 * hsam_canon / 5);
    let _ = hprintln!("baseline recency (LRU):     {}/5 = {}%", recency_canon, 100 * recency_canon / 5);
    let _ = hprintln!("baseline frequency (smart): {}/5 = {}%", freq_canon, 100 * freq_canon / 5);
    let _ = hprintln!("nodes_after={} canon_after={}", nexus.len(), nexus.canon_count());

    // ── Part 2: does the eviction ORDER itself do anything? ──────────────────────────────
    //
    // Part 1 only proves the canon exemption: it returns 5/5 even with a broken ordering
    // among ephemeral nodes, so on its own it is not evidence that the policy works. This
    // part isolates the ordering. Every node here is ephemeral and equally unremarkable to
    // recency and frequency heuristics — the ONLY thing separating them is what a human said.
    //
    // The confirmed facts are deliberately the OLDEST and least accessed, so recency and
    // frequency baselines would both throw exactly them away.
    let mut n2 = MemoryGraphNexus::new();
    let mut confirmed: Vec<String> = Vec::new();
    for i in 0..10 {
        let mut c = String::from("human-confirmed fact ");
        c.push_str(itoa(i).as_str());
        confirmed.push(n2.create_node(c, String::new(), Vec::new(), SourceType::UserUtterance, 2, None, CanonLevel::None));
    }
    let mut rejected: Vec<String> = Vec::new();
    for i in 0..10 {
        let mut c = String::from("human-rejected fact ");
        c.push_str(itoa(i).as_str());
        rejected.push(n2.create_node(c, String::new(), Vec::new(), SourceType::UserUtterance, 2, None, CanonLevel::None));
    }
    for i in 0..10 {
        let mut c = String::from("untouched fact ");
        c.push_str(itoa(i).as_str());
        n2.create_node(c, String::new(), Vec::new(), SourceType::UserUtterance, 2, None, CanonLevel::None);
    }
    // Newer + heavily used nodes are the ones a human called wrong: the heuristics love them.
    for id in &rejected {
        for _ in 0..8 {
            n2.record_access(id);
        }
        n2.record_human_feedback(id, false);
    }
    for id in &confirmed {
        n2.record_human_feedback(id, true);
    }

    // What recency/frequency WOULD keep, computed before eviction.
    let (recency_kept, freq_kept) = {
        let all = n2.get_node();
        let mut v: Vec<(bool, u64, u32)> = all
            .values()
            .map(|n| (confirmed.iter().any(|c| *c == n.id), n.last_accessed_at, n.access_count))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        let rk = v.iter().take(10).filter(|x| x.0).count();
        v.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)));
        let fk = v.iter().take(10).filter(|x| x.0).count();
        (rk, fk)
    };

    n2.enforce_capacity(10);
    let hsam_kept = confirmed.iter().filter(|id| n2.get_node().contains_key(*id)).count();
    let rejected_left = rejected.iter().filter(|id| n2.get_node().contains_key(*id)).count();

    let _ = hprintln!("=== eviction ORDER among ephemeral nodes (30 -> 10, no canon) ===");
    let _ = hprintln!("human-confirmed facts kept: HSAM {}/10 = {}%", hsam_kept, 10 * hsam_kept);
    let _ = hprintln!("  baseline recency (LRU):   {}/10 = {}%", recency_kept, 10 * recency_kept);
    let _ = hprintln!("  baseline frequency:       {}/10 = {}%", freq_kept, 10 * freq_kept);
    let _ = hprintln!("human-rejected facts still present: {}/10", rejected_left);
    let _ = hprintln!("nodes_after={}", n2.len());

    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}

/// Minimal usize -> String (no_std, no format machinery needed here).
fn itoa(mut n: usize) -> String {
    if n == 0 {
        return String::from("0");
    }
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    String::from_utf8_lossy(&buf[i..]).into_owned()
}
