---
name: mojo-gpu-kernel
description: Conventions for writing, building, testing, and launching Mojo GPU code in this repo — harness entry points that call MAX kernels, the bandwidth probe, and (later phases) our own kernels in kernels/. Covers dealing with Mojo API drift, DeviceContext/LayoutTensor usage, compile-time dispatch by GPU architecture, correctness tests with per-format tolerances, and the required order correctness → timing → profile. Use this whenever you write or edit a .mojo file, hit a Mojo compile error, launch a GPU kernel, or set up a Mojo test.
---

# Mojo GPU code in this repo

## Rule zero: the installed source is the API reference

Mojo's GPU API changes between monthly releases. Whatever you remember about
`DeviceContext`, `DeviceBuffer`, `LayoutTensor`, `gpu.mma`, `async_copy`,
`barrier`, keyword names, or import paths is a hypothesis. Before writing code
that uses an API:

```
pixi run mojo --version
STD=modular/mojo/stdlib/std          # adjust to what exists
grep -rn "struct DeviceContext" $STD/gpu/host/
grep -rn "fn enqueue_function\|fn synchronize\|fn create_buffer\|fn enqueue_copy" $STD/gpu/host/device_context.mojo
grep -rn "struct LayoutTensor" modular/max/kernels/src/layout/ 2>/dev/null || grep -rn "struct LayoutTensor" $STD
```

Known drift as of the repo's creation (verify): `alias` → `comptime`;
`from gpu import ...` → `from std.gpu import ...`; stdlib lives under `std.`.
Every drift you fix goes into `reports/api-drift.md` as `old → new (mojo
version)`.

When a MAX kernel's signature is unclear, read its existing test under
`modular/max/kernels/test/` — tests are the most reliable usage examples.

## Building and running

- `pixi run mojo run bench/mojo/<entry>.mojo -- <args>` for quick runs.
- `pixi run mojo build` for anything you will launch 30× in a sweep, so
  compile time is not inside the measurement window (it isn't anyway, but keep
  the sweep script honest).
- `pixi run mojo test tests/mojo/` for Mojo tests. Python-side tests via
  `uv run pytest`.
- Kernels that import from `modular/max/kernels/src` need the import path set;
  check how upstream tests do it (`-I` flags, `pixi.toml` tasks, or a
  `mojoproject`-style config) and copy that into `pixi.toml` tasks rather than
  ad-hoc flags.

## Structure of a harness entry point (`bench/mojo/`)

One file per (kernel, impl). Each does exactly this, in order:

1. Parse args: shape, format, n_samples, launches_per_sample, seed,
   `--validate-only`, `--out <json>`.
2. Allocate host inputs with the seeded generator from `bench/mojo/inputs.mojo`
   (same seed → same bytes as the Python baselines; that is what makes
   cross-implementation validation possible).
3. Upload, launch **once**, download, compare against the reference produced by
   `bench/reference.py` (written to disk as `.npy`; the Mojo side reads it
   rather than recomputing, so all impls are validated against one reference).
4. Only if validation passes: warmup, timed samples per `bench-methodology`.
5. Write the results JSON through the shared writer (call the Python writer via
   subprocess with a temp file, or write the JSON directly matching the schema
   exactly — do not drift the schema).

Keep host/device plumbing in `bench/mojo/common.mojo`; entry points should be
short.

## Correctness tolerances

Reference = fp32 on host: dequantize weights exactly per format, then matmul in
fp32 with fp32 activations cast from the fp16 inputs. Compare the kernel's fp16
or fp32 output against that.

| Weights | Activations | Accum | rtol | atol | Notes |
|---|---|---|---|---|---|
| fp16 | fp16 | fp32 | 1e-2 | 1e-3 | |
| Q8_0 | fp16 | fp32 | 1e-2 | 1e-3 | dequant is exact; error is accumulation order |
| Q4_0 / Q4_K | fp16 | fp32 | 1e-2 | 2e-3 | |
| any | fp16 | fp16 accum | 3e-2 | 5e-3 | flag fp16 accumulation in the JSON; llama.cpp uses it in some paths |
| attention decode, fp16 KV | | fp32 | 2e-2 | 2e-3 | compare against fp32 reference softmax(QKᵀ/√d)V |

Report max abs and max rel error in the results JSON. A kernel that passes
within tolerance but has an error 10× larger than a sibling implementation is
worth a note.

## Compile-time dispatch by architecture (for our kernels, later phases)

Follow the pattern already proven in `Cro22/mojo-cuda-ampere`: warp size, tile
dims, num_stages, and smem budget are `comptime` parameters selected from the
device info at compile time, with an explicit table for sm_75 / sm_86 / sm_89
and a documented fallback. Never branch on architecture at runtime inside the
kernel. Keep the table in one file (`kernels/config.mojo`) so a Modular
reviewer can see every consumer-specific decision in one place — that file is
part of what we would upstream.

Constraints to design around on sm_86/sm_89: 100 KB configurable shared memory
per SM, 64 K registers per SM, `mma.sync` m16n8k16 only (no wgmma, no TMA),
`cp.async` available for multi-stage pipelining. The 4090 has 128 SMs vs 82:
grid sizing that saturates the 3090 under-fills the 4090 for small N — check
occupancy on both.

## Order of operations for any kernel work

correctness test → benchmark under locked clocks → nsys confirm → ncu only if
% roofline is below expectation → optimize → repeat. Never reorder. Never
optimize a kernel whose test is not green.

## Things that have wasted time before

- Timing that includes the H2D upload. Upload once, outside the loop.
- Comparing fp16 output to an fp16 reference (double rounding). Reference is
  always fp32.
- Forgetting that Q4_0 stores element j in the low nibble of byte j and element
  j+16 in the high nibble of the same byte (see `gguf-quant-formats`).
- Assuming a MAX kernel's "M" is rows of the activation; check whether the
  upstream matmul is `C = A·B` or `C = A·Bᵀ` (weights stored N×K vs K×N) before
  building shapes.
