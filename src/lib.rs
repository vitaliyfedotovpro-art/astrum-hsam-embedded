#![cfg_attr(not(any(test, feature = "std")), no_std)]

extern crate alloc;

// Astrum HSAM — Hypergraph Sparse Associative Memory (Rust native core)
// Hypergraph memory engine with provenance-weighted recall, canon retention,
// affect overlay, and a 24-cell cognitive ontology.

/// Global allocator for `no_std` builds (delegates to system malloc/free).
/// Only active when building without std; under `cargo test` (std) the
/// standard library provides the allocator.
#[cfg(feature = "runtime")]
mod system_alloc {
    use core::alloc::{GlobalAlloc, Layout};

    extern "C" {
        fn malloc(size: usize) -> *mut u8;
        fn posix_memalign(memptr: *mut *mut u8, alignment: usize, size: usize) -> i32;
        fn free(ptr: *mut u8);
        fn realloc(ptr: *mut u8, size: usize) -> *mut u8;
    }

    struct SystemAlloc;

    unsafe impl GlobalAlloc for SystemAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if layout.align() > 8 {
                let mut ptr: *mut u8 = core::ptr::null_mut();
                let _ = posix_memalign(&mut ptr, layout.align(), layout.size());
                ptr
            } else {
                malloc(layout.size())
            }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
            free(ptr);
        }

        unsafe fn realloc(&self, ptr: *mut u8, _layout: Layout, new_size: usize) -> *mut u8 {
            realloc(ptr, new_size)
        }
    }

    #[global_allocator]
    static ALLOCATOR: SystemAlloc = SystemAlloc;
}

/// Panic handler required for `no_std` builds. Simply loops — the host
/// environment is expected to provide its own fault handling.
#[cfg(feature = "runtime")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

pub mod affect_overlay;
pub mod c_api;
pub mod canon;
pub mod graph_nexus;
pub mod persistence;
pub mod provenance;
pub mod ricci_curvature;
pub mod topology_24cell;
pub mod vector_index;

pub use affect_overlay::{AffectOverlayEngine, AffectState};
pub use c_api::*;
pub use canon::CanonLevel;
pub use graph_nexus::{HyperEdge, MemoryEdge, MemoryGraphNexus, MemoryNode};
pub use persistence::Snapshot;
pub use provenance::{PfcAction, PfcGuard, SourceType};
pub use ricci_curvature::FormanRicciCalculator;
pub use topology_24cell::{StateCell, Topology24Cell};
pub use vector_index::{
    cosine_similarity, CapiIndex, Int8VectorIndex, SimpleVectorIndex, VectorSearchResult,
    CAPI_INDEX_KIND,
};
