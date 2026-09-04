# Astrum HSAM — embedded

[![CI](https://github.com/vitaliyfedotovpro-art/astrum-hsam-embedded/actions/workflows/ci.yml/badge.svg)](https://github.com/vitaliyfedotovpro-art/astrum-hsam-embedded/actions/workflows/ci.yml)

A `no_std` Rust memory engine for AI agents that run on small hardware. It does two things a
vector store does not:

- **Keeps a model's own output from becoming its evidence.** Self-descriptions are quarantined
  from recall outright, not down-ranked. Measured on Cortex-M4: contamination **66.6% → 0%**.
- **Keeps critical facts alive under memory pressure.** Safety rules and operator constraints
  are exactly the rarely-read items an LRU policy drops first. Measured: **100% retained**,
  against 0% for both recency and frequency baselines.

Around 35 KB of code, ~800 bytes per stored fact, no cloud calls, no runtime dependencies.
Cross-compiles to Cortex-M4 and RISC-V; a C-ABI is included.

## Start here

| | |
|---|---|
| **[MATURITY.md](MATURITY.md)** | **What this is not.** Research prototype: no hardware testing, no fuzzing, no soak, and running out of heap hangs the device. Read before anything else. |
| [RESULTS.md](RESULTS.md) | Every measurement, with the method and the command to re-run it |
| [INTEGRATION.md](INTEGRATION.md) | Building it, feeding it, the rules that bite if ignored |
| [`include/astrum_memory.h`](include/astrum_memory.h) | The C API |
| [`ctest/main.c`](ctest/main.c) | Shortest working example — 32 checks against the real library |

## In sixty seconds

```bash
export PATH=~/.rustup/toolchains/stable-*/bin:$PATH   # rustup's rustc must win over Homebrew's
cargo test                                            # 29 tests
cargo rustc --release --features std --crate-type staticlib
cc ctest/main.c -Iinclude -Ltarget/release -lastrum_memory -o ctest/ctest && ./ctest/ctest
```

The QEMU benchmarks need `qemu-system-arm`; every command is listed in
[INTEGRATION.md §6](INTEGRATION.md).

CI runs the rest on every push: both test suites, `clippy -D warnings`, `rustfmt`, the two
cross-compile targets, the C test under ASan + UBSan, and Miri with strict provenance. The
QEMU benchmarks are not in CI — they report numbers rather than pass/fail, and pinning them
needs a regression tolerance that has not been agreed.

## The one-paragraph version of the design

Facts carry **provenance** — who said this: the user, an external document, the model itself.
Provenance sets a ceiling on how important a fact may become and decides whether it is eligible
for recall at all. Facts may be marked **canon**, which makes them immune to eviction however
old or rarely read. A human can confirm or reject a recall, and that verdict moves how long a
fact survives — never how it ranks, because folding value into relevance was measured to cost
86% → 29% recall@1. Retrieval is cosine similarity weighted by provenance, over an exact linear
scan, with an optional int8 index that trades ~16% query time for 59% of the memory.

No signal derived from the engine's own behaviour is ever treated as evidence. That restriction
is the product.

## Status

Prototype under active development, single author. The measurements are honest and
reproducible; the code around them has never run on real hardware. The sensible next step with
a partner is a joint evaluation on their target, not a binary drop into a product — see
[MATURITY.md](MATURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
