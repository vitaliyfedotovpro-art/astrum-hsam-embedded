# Integrating Astrum HSAM

What you get: a `no_std` Rust memory engine with a C-ABI, ~35 KB of code, no runtime
dependencies, no network calls. What it does that a vector store does not: **quarantines a
model's own output from recall** and **pins critical facts through eviction**. Numbers behind
both claims, measured on Cortex-M4 under QEMU: [RESULTS.md](RESULTS.md).

Shortest working example: [`ctest/main.c`](ctest/main.c). API reference:
[`include/astrum_memory.h`](include/astrum_memory.h).

**Before you integrate, read [MATURITY.md](MATURITY.md).** This is a research prototype: no
hardware testing, no fuzzing, no soak run, and heap exhaustion hangs the device. The page lists
every known limitation rather than leaving you to find them.

---

## 1. Which stack are you building?

**Embedded Linux (Ambarella, Rockchip, Raspberry Pi, any board with an MMU and a libc).**
The full stack works today: a local sentence embedder (Candle/MiniLM) produces vectors, HSAM
stores and ranks them, everything stays on the device. RAM is not the constraint here — the
exact f32 index is fine (~1.9 KB per fact; 10,000 facts ≈ 19 MB).

**Bare-metal MCU (Cortex-M, RISC-V).** HSAM runs, but it is memory *middleware only*: there is
no embedder on the device. `src/embed.rs` in the reference desktop crate is Candle-based and
desktop-only; it is not part of this port. **You supply the vectors** — from your NPU/DSP/CIM
block, from a host over a link, or precomputed at provisioning time. HSAM never calls a model.
Here the int8 index earns its keep (801 vs 1,949 bytes per fact).

Either way HSAM is the memory layer, not the model. It takes vectors in and ranks facts out.

---

## 2. Allocator and panic handler

The crate is `no_std` + `alloc`. Someone has to provide a global allocator and a panic
handler — either you or the crate, never both.

**You provide them** (normal for bare metal). Then consume HSAM as a **Cargo dependency of your
firmware crate**, not as a prebuilt `.a` — a standalone staticlib has nowhere to get an
allocator from and rustc refuses to build one ("no global memory allocator found",
"`#[panic_handler]` function required"). In your firmware's `Cargo.toml`:

```toml
astrum_memory = { path = "../astrum-hsam-embedded", package = "astrum-hsam-embedded",
                  default-features = false, features = ["capi-int8"] }
```

and in the firmware itself:

```rust
use embedded_alloc::LlffHeap as Heap;

#[global_allocator]
static HEAP: Heap = Heap::empty();

const HEAP_SIZE: usize = 1024 * 1024;              // size it from RESULTS.md
static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];

#[entry]
fn main() -> ! {
    unsafe { HEAP.init(core::ptr::addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE); }
    // ... engine calls
}
```

A complete working version is [`footprint-fw/src/main.rs`](footprint-fw/src/main.rs) —
that firmware is what produced the measured numbers.

**The crate provides them**: build with `--features runtime`. This is what makes a standalone
`no_std` staticlib possible — the allocator and panic handler live inside the `.a`, so a C build
system can link it like any other library. Do not combine it with your own `#[global_allocator]`;
the build will refuse.

What `runtime` actually installs is a *shim*, not a heap. It allocates nothing itself — it calls
four ordinary C symbols, which the library leaves undefined for your linker to resolve:

```
$ nm -u libastrum_memory.a | grep -E 'malloc|free|realloc|memalign'
U malloc    U free    U realloc    U posix_memalign
```

So a **full libc is not required** — newlib, the FreeRTOS heap, or your own allocator over a
static array all work, as long as those four symbols exist at link time (`posix_memalign` is
only reached for alignments above 8; a stub returning an aligned block is enough). The panic
handler in `runtime` is a spin loop: it stops the fault, it does not report it. If you need a
reset or a log on panic, take the crate as a Cargo dependency and write your own.

The one combination that cannot exist is **a prebuilt `.a` plus your own Rust
`#[global_allocator]`** — that wiring happens when the Rust code is compiled, not when C links
it. Supply your heap as C symbols (above), or build HSAM into your firmware (§2 first option).

Sizing the heap: bytes/node × expected facts, plus headroom. 384-dim embeddings cost
**1,949 B/node** (f32) or **801 B/node** (int8) including graph, provenance and string overhead.
Allocation is linear in node count; the engine has ~0 static RAM.

**Running out of heap hangs the device — plan for it.** Measured, not assumed (RESULTS.md,
`oom.rs`): an allocation failure panics, and the `runtime` panic handler is a spin loop, so an
exhausted engine stops responding instead of shedding load. There is no fallible-allocation
API to catch it. The mitigation is to stay inside a budget rather than discover the ceiling:

```c
if (astrum_memory_node_count(mem) > BUDGET) {
    astrum_memory_enforce_capacity(mem, BUDGET);   /* frees embeddings too */
}
```

The same firmware shows this working: 200 inserts against a 25-node budget on a 64 KiB heap
held at a constant 53 KB and never grew, while the unbounded version died at ~32 nodes.
Teardown is clean — dropping the engine returned the heap to zero, so the budget is the only
thing you have to get right. If you need a reset or a fault log instead of a spin, take the
crate as a Cargo dependency and supply your own `#[panic_handler]`.

---

## 3. Building the library

```bash
# rustup's toolchain must come FIRST in PATH. A Homebrew rustc has no target std and
# fails with "can't find crate for core" — a confusing error with a boring cause.
export PATH=~/.rustup/toolchains/stable-*/bin:$PATH

# Host, with file persistence (astrum_memory_save/load available)
cargo rustc --release --features std --crate-type staticlib

# libc host, no_std, crate-provided allocator
cargo rustc --release --no-default-features --features runtime --crate-type staticlib

# Cortex-M4F, int8 index, crate-provided allocator shim (your malloc/free at link time)
cargo rustc --release --no-default-features --features runtime,capi-int8 \
      --target thumbv7em-none-eabihf --crate-type staticlib

# RISC-V (rv32imac), same
cargo rustc --release --no-default-features --features runtime,capi-int8 \
      --target riscv32imac-unknown-none-elf --crate-type staticlib
```

Output: `target/<triple>/release/libastrum_memory.a`. Link it with
`include/astrum_memory.h`. Verified on rustc 1.97.1 for both triples above.

`runtime` expects `malloc`/`free`/`realloc`/`posix_memalign` from your side at link time — any
heap will do (§2). If you would rather supply a Rust allocator, drop `runtime` and take the crate
as a Cargo dependency instead — that path needs no staticlib at all.

Consuming it from Rust instead of C is simpler — add the crate as a path/git dependency and
skip the C-ABI entirely; the Rust API additionally exposes graph traversal
(`associative_recall`, `link_semantic`, Ricci pruning), which the C-ABI does not.

---

## 4. f32 or int8

| | default (f32) | `--features capi-int8` |
|---|---|---|
| bytes/node (384-dim) | 1,949 | **801** |
| facts in 1.4 MB | ~700 | **~1,750** |
| retrieval precision | exact reference | −0.3 pp or less (measured) |

Pick int8 when SRAM is the binding constraint, f32 when it is not. The measured cost of int8
is ≤0.3 percentage points of task precision, and it does **not** amplify error on hard queries
— quality there is set by your embedder, not by quantization ([RESULTS.md](RESULTS.md)).

Two consequences worth knowing before you commit:

- **Snapshots are not interchangeable.** A file written by an f32 build is refused by an int8
  build and vice versa; the file records which one wrote it (`"index_kind": "f32" | "int8"`).
  Through the C-ABI a refused load surfaces as `astrum_memory_load` returning `NULL` — if you
  need the reason, the JSON header names the kind, or use the Rust API where the error text is
  returned.
- **The choice is made at build time**, not at runtime. If you need both in one binary, say so
  — the index types are siblings behind one alias (`CapiIndex` in `src/vector_index.rs`) and a
  runtime switch is a small change, deliberately not made until someone needs it.

---

## 5. Using the API

Lifecycle, in the order a consumer calls it:

```c
AstrumMemoryHandle *mem = astrum_memory_create();

char *id = astrum_memory_add_node(mem, "the user is allergic to penicillin", "medical",
                                  ASTRUM_SOURCE_USER_UTTERANCE, 2,
                                  ASTRUM_CANON_L2_FOUNDATIONAL,  /* never evicted */
                                  embedding, 384);
astrum_memory_free_string(id);

char *json = astrum_memory_search(mem, query, 384, 2, 5);   /* JSON array, ranked */
/* ... parse ... */
astrum_memory_free_string(json);

astrum_memory_destroy(mem);
```

Three fields carry the behaviour that distinguishes this engine — get them right and the rest
is a vector store:

- **`source_type`** — where the fact came from. Mark model-generated self-description as
  `ASTRUM_SOURCE_LLM_SELF_DESCRIPTION` and it is excluded from recall outright, not merely
  down-ranked. This is the anti-self-echo mechanism; feeding everything in as
  `USER_UTTERANCE` disables it.
- **`canon_level`** — `L1`/`L2` facts survive capacity pressure regardless of age or access
  count. Safety rules, clinical protocol, user constraints belong here.
- **`embedding`** — your vectors, your dimension. The first one added fixes the dimension for
  the engine's lifetime; later embeddings of a different length are silently not indexed.
  Pass `NULL` to store a fact without making it searchable by similarity.

### Feedback from the person, not from the loop

The engine accepts one reward signal, and only from a human:

```c
astrum_memory_record_feedback(mem, node_id, 1);   /* that recall was right */
astrum_memory_record_feedback(mem, node_id, 0);   /* that recall was wrong */

/* later, deliberately: make what has been confirmed twice eviction-proof */
astrum_memory_promote_confirmed(mem, 2, ASTRUM_CANON_L1_PROJECT);

/* under memory pressure: least valuable go first, embeddings freed with them */
astrum_memory_enforce_capacity(mem, 2000);
```

**Do not wire this to your agent loop.** "The model retrieved it, so it was useful" is a
signal derived from the model's own output — the same self-echo the provenance guard exists to
break, only harder to notice, and it compounds: the more a fact is recalled the more it is
rewarded for being recalled. The verdict has to come from a person looking at the answer.

Two properties are worth relying on:

- **Feedback moves retention, never ranking.** Search order is a pure function of cosine and
  provenance; a confirmed fact does not float to the top, it survives pressure longer. This is
  not caution, it is a measurement: folding value multipliers into the rank in this codebase
  cost 86% → 29% recall@1, because an important-but-off-topic fact outranked the on-topic one.
- **Provenance is a ceiling that applause cannot lift.** Importance is clamped per source type,
  so no amount of confirmation promotes model self-description to the standing of the user's
  own words, and `promote_confirmed` refuses it outright.

Rejection weighs more than confirmation (−0.20 against +0.15): a wrong recall costs the user
more than a right one gains. Canon is not touched by feedback in either direction — it is a
human decision already, and `promote_confirmed` is how a confirmed fact becomes one.

Rules that will bite if ignored:

- Every `char *` returned by the API is yours and must go to `astrum_memory_free_string()` —
  not `free()`.
- One handle is internally serialized (spinlock), so threads are safe. It is **not
  interrupt-safe**: never call from an ISR on bare metal.
- `astrum_memory_search` returns `"[]"`, never `NULL`, for a valid handle with no match.

---

## 6. Verifying it yourself

Everything below runs from the crate root and should be reproduced before you trust any
number in the pitch:

```bash
cargo test                       # 29 tests, exact f32 build
cargo test --features capi-int8  # same 29, int8 build

# link the header against the library and exercise the engine end to end (32 checks)
cargo rustc --release --features std --crate-type staticlib
cc ctest/main.c -Iinclude -Ltarget/release -lastrum_memory -o ctest/ctest && ./ctest/ctest

# the same C test with memory errors and UB instrumented
cc -g -fsanitize=address,undefined ctest/main.c -Iinclude \
   -Ltarget/release -lastrum_memory -o ctest/ctest_asan && ./ctest/ctest_asan

# the unsafe C-ABI under an interpreter that watches every memory access.
# Needs a nightly toolchain (`rustup toolchain install nightly --component miri`); slow (~4 min).
# --no-default-features drops libm's inline asm, which Miri cannot execute.
PATH=~/.rustup/toolchains/nightly-*/bin:$PATH \
MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-strict-provenance" \
  cargo miri test --no-default-features --features std

# host benches
cargo run --release --example latency        # query cost and how it scales
cargo run --release --example recall_int8    # int8 vs f32 recall
cargo run --release --example graph_lift     # associative recall vs flat

# hardware-representative measurements in QEMU (needs qemu-system-arm)
cd footprint-fw
cargo run --release                          # f32 vs int8 bytes/node, Rust path
cargo run --release --bin capi_footprint     # bytes/node through the C-ABI
cargo run --release --bin contamination      # self-echo 66.6% -> 0%
cargo run --release --bin canon              # canon retention + eviction order
cargo run --release --bin latency            # query cost scaling on the target ISA
cargo run --release --bin oom                # what the heap ceiling does (it hangs)
```

QEMU note: `mps2-an386` has 4 MB of RAM, so the firmware heap is capped at 3 MB. Asking for
more in `memory.x` puts the stack in unbacked memory and the board hard-faults silently.

---

## 7. What is not here

Stated plainly so nobody discovers it late:

- **No embedder on MCU.** Vectors come from you.
- **No graph API in C.** `associative_recall` and the linking/pruning functions are Rust-only
  today; the C-ABI exposes flat provenance-weighted recall. The measured 0% → 100% graph lift
  is reachable from Rust, not from C.
- **No ANN index.** Search is an exact linear scan — fine to tens of thousands of vectors on
  a real CPU, and the honest bound for an MCU is the RAM budget above, not query time.
- **Persistence needs `std`.** `save`/`load` are absent from `no_std` builds; on bare metal,
  serialize with `Snapshot::to_json_vec()` and write the bytes to your own flash driver.
