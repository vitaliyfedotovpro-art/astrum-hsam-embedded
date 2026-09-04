> **Read [MATURITY.md](MATURITY.md) alongside this file.** Every number here is real and
> re-runnable, which is a statement about the measurements — not about the maturity of the code
> around them. This is a research prototype: no hardware testing, no fuzzing, no soak, and
> running out of heap hangs the device.

# Astrum HSAM — bare-metal footprint (MEASURED, not estimated)

Hardware-representative measurement of the `no_std` port on **Cortex-M4 (thumbv7em-none-eabihf)**
under **QEMU `mps2-an386`**, heap accounted by `embedded-alloc` (`LlffHeap::used()`).

## Method
- `no_std` + `alloc` port of `astrum_memory` (this crate), rustc 1.97.1.
- Firmware (`footprint-fw/`) inserts N nodes into `MemoryGraphNexus` + `SimpleVectorIndex`,
  each node carrying a **384-dim f32** embedding (typical small sentence-embedder, e.g. MiniLM).
- Heap `used()` read at peak (structures alive). Realistic short fact strings as content.

## Heap footprint (384-dim f32 embeddings)

| N nodes | heap bytes | bytes/node |
|--------:|-----------:|-----------:|
| 128 | 255,064 | 1,992 |
| 256 | 503,240 | 1,965 |
| 512 | 999,544 | 1,952 |
| 1000 | 1,949,752 | 1,949 |

**Linear. Measured slope ≈ 1,949 bytes/node** (dominated by the 384×4 = 1,536-byte embedding;
the graph/provenance/BTree overhead is ~400 B/node on top).

### Extrapolation (linear, measured slope)
| nodes | heap SRAM |
|------:|----------:|
| ~700 | ~1.36 MB |
| 2,000 | ~3.9 MB |
| 3,000 | ~5.85 MB |

## Static footprint (whole firmware, incl. engine + rt + serde_json + libm)
- `.text` ≈ **35 KB** flash (code)
- `.data` = 0, engine static RAM ≈ 0 (`.bss` shown is only the reserved 3 MiB heap buffer)

## Honest conclusions
1. The pitch figure **"~1.4 MB SRAM"** holds only **~700 nodes at 384-dim f32** — NOT "thousands".
   Real cost is **~1.95 KB/node** at this dim/precision.
2. To fit thousands of nodes in ~1.4 MB you MUST reduce the embedding cost:
   - **int8 quantization**: 384 B vs 1,536 B → ~800 B/node → 1.4 MB ≈ **~1,750 nodes**;
   - **smaller dim** (e.g. 128-dim int8 = 128 B) → ~500 B/node → 1.4 MB ≈ **~2,800 nodes**.
   These are code changes not yet made (`SimpleVectorIndex` stores `Vec<f32>` today).
3. Code footprint is tiny (~35 KB flash, ~0 static RAM); SRAM is essentially all embedding data.
4. Caveat: measured on a monotonic-insert workload; prune/evict churn may add heap fragmentation.

## Reproduce
```
# no_std proof + tests (host), both index backends
cargo test
cargo test --features capi-int8
cargo build --no-default-features --features runtime --release   # no_std staticlib links via libc

# cross-compile the C-ABI staticlib deliverable (Cortex-M4F), int8 backend
PATH=~/.rustup/toolchains/stable-*/bin:$PATH \
cargo rustc --release --no-default-features --features runtime,capi-int8 \
      --target thumbv7em-none-eabihf --crate-type staticlib

# link the public header against the library and exercise the engine from C
cargo rustc --release --features std --crate-type staticlib
cc ctest/main.c -Iinclude -Ltarget/release -lastrum_memory -o ctest/ctest && ./ctest/ctest

# hardware-representative heap measurement in QEMU
cd footprint-fw && cargo run --release
```

Integration guide (allocator, targets, API rules, limitations): [INTEGRATION.md](INTEGRATION.md).

---

# int8 vs f32 recall (examples/recall_int8.rs)

Same graph, N=2000, D=384, 50 clusters. Ground truth = exact f32 top-k.
int8 = per-vector symmetric quantization (D+4 B/vec vs 4D). Query kept f32 (asymmetric).

| sigma | task | fidelity@5 (int8 vs exact-f32) | f32 prec@5 | int8 prec@5 | delta |
|------:|------|------------------------------:|-----------:|------------:|------:|
| 0.03 | easy | 98.6% | 100.0% | 100.0% | +0.00 |
| 0.06 | easy | 99.2% | 100.0% | 100.0% | +0.00 |
| 0.10 | easy | 99.2% | 100.0% | 100.0% | +0.00 |
| 0.15 | med  | 99.1% |  77.3% |  77.0% | -0.27 pp |
| 0.20 | hard | 99.1% |  31.3% |  31.3% | -0.07 pp |

**int8 costs <=0.3 pp on the task at 4x vector-memory savings**, and does NOT amplify error
when the task is hard (task difficulty, not quantization, drives the drop). Quantization
touches only the fuzzy nearest-neighbor entry step — graph/provenance/canon are exact.

## int8 index — MEASURED end-to-end in QEMU (now wired into the engine)

`Int8VectorIndex` (i8 codes + per-vector scale; scale cancels in cosine, no dequant/alloc)
is a real sibling type in the crate (f32 index stays the exact reference). Same firmware,
same graph, measured on Cortex-M4:

| N | f32 B/node | int8 B/node | saved |
|--:|-----------:|------------:|------:|
| 128 | 1992 | 844 | 58% |
| 256 | 1965 | 817 | 59% |
| 512 | 1952 | 804 | 59% |
| 1000 | 1949 | **801** | 59% |

**int8 = ~801 B/node measured** (matches the ~800 prediction). Capacity per SRAM budget:

| | f32 | int8 |
|---|---:|---:|
| 1.4 MB | ~700 nodes | **~1,750 nodes** |
| 2,000 nodes | 3.9 MB | **1.6 MB** |
| 3,000 nodes | 5.85 MB | **2.4 MB** |

## C-ABI path — what a linked `.a` actually costs (footprint-fw/src/bin/capi_footprint.rs)

The tables above insert through the Rust API. A partner links the **C-ABI**, so that path was
measured separately: every node goes in via `astrum_memory_add_node` (content as a C string,
returned node id freed by the caller), recall via `astrum_memory_search`. Built with
`--features capi-int8`, i.e. the C-ABI backed by `Int8VectorIndex`.

| N | heap bytes | B/node |
|--:|-----------:|-------:|
| 128 | 103,480 | 808 |
| 256 | 200,104 | 781 |
| 512 | 393,304 | 768 |
| 1000 | 765,816 | **765** |

**The shippable path is not more expensive than the headline — it is slightly cheaper (765 vs 801).**
The gap is harness, not engine, and was measured rather than assumed:
- ~16 B/node: the Rust harness reads `HEAP.used()` while the query's result buffer is still alive
  (`Vec<VectorSearchResult>` keeps capacity N after `truncate`). Freeing it before the read gives
  785 B/node int8 / 1933 f32 — a one-line probe, reverted so the published numbers stay as measured.
- the remainder: `format!` leaves spare capacity in the two content strings, while strings crossing
  the C boundary are allocated at exact length.

**801 B/node therefore stands as the conservative published figure.** Default builds (no
`capi-int8`) put the exact f32 index behind the C-ABI instead — ~1,949 B/node; the feature is what
makes the shipped binary match the headline.

Reproduce: `cd footprint-fw && cargo run --release --bin capi_footprint`

# Query latency (examples/latency.rs, footprint-fw/src/bin/latency.rs)

Footprint says what a fact costs to store; this says what a recall costs to answer. Search is
an exact linear scan, so the expectation is time linear in N x D — measured, not assumed.

## Desktop wall clock (D=384, top_k=5, 200 queries/point)

| N | f32 us/query | int8 us/query | f32 ns/node | int8 ns/node |
|--:|-------------:|--------------:|------------:|-------------:|
| 128 | 60 | 70 | 472 | 546 |
| 512 | 149 | 132 | 291 | 258 |
| 2,000 | 397 | 463 | **199** | **232** |
| 10,000 | 1,990 | 2,320 | 199 | 232 |

Flat ns/node above ~2k confirms the linear scan; the higher figure at N=128 is fixed per-query
overhead amortising out. Spread across three runs was under 2%.

**int8 is a MEMORY win, not a speed win** — it is reproducibly ~16% SLOWER than f32 here
(i8→f32 conversion per element, and the f32 path vectorises better). Do not claim int8 is
faster anywhere; it buys 59% of the RAM back and costs a little time.

## Cortex-M4 (QEMU) — scaling only

| N | f32 ticks/query | ticks/node | int8 ticks/query | ticks/node |
|--:|----------------:|-----------:|-----------------:|-----------:|
| 128 | 16,768 | 131 | 14,466 | 113 |
| 512 | 68,451 | 133 | 71,187 | 139 |
| 1,000 | 133,650 | 133 | 123,143 | 123 |

Ticks per node stay flat, so the scan is linear on the target ISA too. **These are QEMU
virtual ticks and are NOT device latency** — QEMU models neither the M4 pipeline, nor flash
wait states, nor real FPU timing. A latency claim needs a real board; until then the honest
statement is the operation count, which is exact: **N x D multiply-accumulates per query**
(1.15M at 3,000 nodes and 384 dims). Multiply by the measured per-MAC cost of the actual part
to get a budget.

## Selection fix found while measuring

The scan used to score every vector into a `Vec` and sort all N, cloning one heap `String` per
stored vector on every query — 3,000 allocations per recall at 3,000 nodes. Replaced with a
bounded top-k shortlist that clones only the k winners' ids.

Honest size of the win: **~17% on desktop (247 → 199 ns/node), not the large speedup the
allocation count suggests** — the scan is compute-bound, not allocation-bound. It is kept for
the heap behaviour on a device, where per-query allocation churn is what fragments the heap
(a risk RESULTS.md flags as untested), not for the arithmetic.

# Heap exhaustion — what happens at the ceiling (footprint-fw/src/bin/oom.rs)

An embedded consumer needs this answered before anything else, so it was run rather than
reasoned about: a 64 KiB heap on Cortex-M4 (QEMU), 384-dim f32, deliberately driven into the
ground.

| | result |
|---|---|
| Bounded (evict to a 25-node budget, 200 inserts) | steady state: 25 nodes, heap **53,388 B constant**, never approaches the ceiling |
| Teardown after the bounded run | heap returns to **0** — no leak from graph, index or strings |
| Unbounded (no budget) | allocator fails at **~32 nodes** (64,148 B of 65,536 used at i=30) |
| Behaviour on failure | allocation failure → `handle_alloc_error` → **panic** |

**The failure mode is a hang, not a degradation.** With `--features runtime` the crate's panic
handler is a spin loop, so an exhausted device stops responding rather than shedding load or
resetting. This is the single most important thing for a consumer to know, and the mitigation
is not subtle: **size the budget from the measured bytes/node and call `enforce_capacity`
before the ceiling, never after.** The bounded row above is that pattern working — the same
engine, the same heap, 200 inserts, no growth.

Consumers who need a reset or a fault log instead of a spin must take the crate as a Cargo
dependency and supply their own `#[panic_handler]` (see INTEGRATION.md §2).

Reproduce: `cd footprint-fw && cargo run --release --bin oom`

# Graph lift on the hard case (examples/graph_lift.rs)

Real engine (`associative_recall`). Target is embedding-distant from the query but joined to a
near-query seed by a co-occurrence edge. 2000 nodes, D=384, top-k=10, 1 hop. Noise edges added
as a stress test.

| noise-edges/node | flat recall | assoc recall | assoc set size | target share |
|-----------------:|------------:|-------------:|---------------:|-------------:|
| 0 | 0.3% | 100.0% | 13.6 | 7.41% |
| 1 | 0.3% | 100.0% | 33.1 | 3.09% |
| 3 | 0.0% | 100.0% | 71.8 | 1.41% |
| 8 | 0.0% | 100.0% | 167.3 | 0.60% |

**Graph turns 0% -> 100% recall on a case flat similarity cannot solve.** Cost: expansion
returns a larger, dirtier candidate set as edge density grows (recall traded for precision).
Levers that keep it tight: deliberate/thresholded edges (not random), Ricci pruning of
redundant edges, and a rerank pass (associative_recall places the target IN the set but does
not rank it — a cosine+provenance rerank pulls it to the top).

## Overall lever map
- Easy tasks / memory size: int8 (4x smaller, ~free quality).
- Medium/hard recall: NOT quantization (f32 itself falls). Levers = better embedder (upstream)
  + HSAM graph edges (associative recall) + rerank. Measured graph lift: 0% -> 100%.

---

# Memory hygiene — MEASURED on Cortex-M4 (QEMU), not desktop

Ported the desktop provenance/canon benchmarks to bare-metal firmware bins
(`footprint-fw/src/bin/contamination.rs`, `canon.rs`), run on QEMU mps2-an386.

## Self-echo contamination@3 (int8 index, 3000 nodes)
200 user "truths", each with 2 high-cosine LLM self-echo copies; 100 canon; filler
LLM-generation to 3000. For each truth, inspect top-3:

| source | flat (plain cosine) | guard (floor + quarantine + rerank) |
|---|---:|---:|
| **self-echo** (LlmSelfDescription) | **66.6%** (400/600) | **0.0%** (0/600) |
| general-LLM (LlmGeneration) | 0.0% | 0.1% (1/600) |

**Self-echo is quarantined to a literal 0/600** (rank_multiplier 0.0 → hard-excluded).
Matches the desktop 67% baseline. The single general-LLM leak is a down-weighted
(not excluded) generation node — honest caveat, not self-echo. heap ≈ 2.43 MB.

## Canon retention under pressure (50 → 10 nodes)
5 critical canon facts (rarely accessed, oldest) + 45 frequently-accessed noise.

| policy | canon survived |
|---|---:|
| **HSAM (canon pinned)** | **5/5 = 100%** |
| recency LRU (keep-newest) | 0/5 = 0% |
| frequency (keep-most-used) | 0/5 = 0% |

Importance is neither recent nor frequent — only the explicit canon flag preserves
rarely-used critical facts. Both naive baselines evict all canon.

### The eviction ORDER, isolated (30 → 10, no canon involved)

The table above tests the canon exemption and nothing else — it returns 5/5 even when the
ordering among ephemeral nodes is broken, so on its own it is weak evidence. This second part
removes canon entirely and varies only what a human said. Confirmed facts are deliberately the
oldest and least accessed, so the heuristics would throw exactly them away.

| policy | human-confirmed facts kept |
|---|---:|
| **HSAM (importance-led eviction)** | **10/10 = 100%** |
| recency LRU | 0/10 = 0% |
| frequency | 0/10 = 0% |

Human-rejected facts still present afterwards: **0/10**.

**This one discriminates, and that was verified by breaking it on purpose:** with the previous
recency-first ordering restored, the same benchmark reports 0/10 confirmed kept and 10/10
rejected still present, while part 1 continues to read a comfortable 5/5. A regression in the
eviction policy now changes a number instead of hiding behind the canon exemption.

## Now every headline number is measured on Cortex-M4 (QEMU):
- footprint 801 B/node (int8), ~1750 facts / 1.4 MB, ~35 KB code — QEMU mps2-an386;
- self-echo contamination 66.6% → 0% — QEMU mps2-an386;
- canon retention 100% vs 0%/0% — QEMU mps2-an386;
- int8 recall <=0.3 pp, graph lift 0%->100% — host (recall is platform-independent).
