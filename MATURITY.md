# Maturity — what this is, and what it is not

**Astrum HSAM (embedded) is a research prototype with reproducible benchmarks.** Not a
product, not a validated component, not something to put in a shipping device on the strength
of a blog post. Every number in [RESULTS.md](RESULTS.md) is real and re-runnable; that is a
statement about the measurements, not about the maturity of the code around them.

This page exists so nobody has to discover the gaps by hitting them.

## Scale of the thing

| | |
|---|---|
| Engine source | ~2,700 lines of Rust, single author |
| Tests | 29 unit tests, both index backends |
| C boundary | 32 checks in `ctest/main.c`, clean under ASAN + UBSan |
| Unsafe code | ~23 raw-pointer sites in `src/c_api.rs`, clean under Miri (strict provenance, leak check) |
| Platforms exercised | host (macOS arm64), Cortex-M4 and RISC-V via QEMU |
| Real hardware | **none** |

## What has actually been checked

- **No undefined behaviour in the C-ABI**: the full test suite runs under Miri with strict
  provenance and leak checking, clean. This covers the handle lifecycle, the spinlock, the
  string hand-off and the caller-supplied embedding slices.
- **No leaks across the boundary**: verified twice, independently — Miri's leak check on host,
  and on Cortex-M4 by tearing the engine down and watching the heap return to zero.
- **Behaviour at the memory ceiling**: measured, and it is a hang (see below).
- **The benchmarks discriminate**: the eviction benchmark was deliberately broken to confirm
  it reports the breakage. The older canon benchmark does not — it is insensitive to eviction
  ordering, and is labelled as such in RESULTS.md.

## Known limitations — read before integrating

**Exhausting the heap hangs the device.** Allocation failure panics, and the `runtime` panic
handler is a spin loop. There is no fallible-allocation API. Stay inside a budget with
`enforce_capacity`; the pattern is in INTEGRATION.md §2 and is measured to work.

**No device latency figure exists.** Query cost was measured on desktop and its *scaling* was
confirmed on Cortex-M4 under QEMU, which is not cycle-accurate. The honest number for a
partner is the operation count (N × D multiply-accumulates per query); anything in
milliseconds must come from real silicon. Do not quote the QEMU ticks as latency.

**Fragmentation under long-running churn is untested.** The footprint numbers come from
monotonic inserts and from insert/evict cycles over minutes, not from days of mixed traffic on
a device. A long soak has never been run.

**Search is an exact linear scan.** No ANN index. Cost is linear in nodes × dimensions, so the
practical ceiling on an MCU is set by your latency budget, not only by RAM.

**No embedder on device.** Vectors come from the consumer's own NPU/DSP or from a host.

**The C-ABI exposes flat recall only.** Graph traversal, semantic linking and Ricci pruning —
including the measured 0% → 100% associative lift — are reachable from Rust, not from C.

**Concurrency is serialised, not concurrent.** One spinlock around the whole engine. It is
correct under threads (checked by Miri) but it is not interrupt-safe and it does not scale;
a busy multi-core consumer will see contention.

**Snapshots are build-specific.** f32 and int8 builds write incompatible files. The file
records which wrote it and a mismatch is refused, but there is no converter.

## What has NOT been done

No fuzzing. No property-based tests. No long-running soak. No power measurement. No test on
physical hardware. No third-party review of the unsafe code. No formal validation of any kind,
and nothing here has been through a regulatory process.

## The honest ask

The right next step with a partner is a **joint evaluation on their target**, not a binary
drop into a product. The benchmarks are there to make that conversation concrete: they show
the mechanism works and they show exactly where it has not been tested. Bugs will surface on
real hardware — that is what the evaluation is for, and it is much cheaper to find them there
than after a design win.

Two real bugs were found in a single afternoon of extending this code — eviction ordering that
made a whole feature inert, and eviction that freed almost no memory. Both are fixed and both
are now covered by tests. The rate at which they appeared is the most honest thing this page
can tell you about how many remain.
