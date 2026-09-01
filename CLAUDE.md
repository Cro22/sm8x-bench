# CLAUDE.md

## What this repo is

A rigorous benchmark workbench for LLM inference kernels written in Mojo, run on
consumer NVIDIA GPUs (RTX 3090 = sm_86, RTX 4090 = sm_89). It measures the
kernels that already ship in Modular's open-source repo (`modular/max/kernels`)
against the best CUDA baselines (llama.cpp, FlashInfer, cuBLAS) and against the
hardware roofline, on hardware Modular does not tune for.

Later phases add our own kernels where the audit finds a gap, with the explicit
goal of upstreaming them to `modular/max/kernels`. This repo is the workbench and
the public record; it is NOT a competing kernel library and must never be framed
as one.

Owner: Jesús (GitHub: Cro22). Prior work with the same methodology:
`github.com/Cro22/mojo-cuda-ampere`. Reuse that methodology, do not reinvent it.

## Current phase: H0 — audit + benchmark of upstream MAX kernels

Definition of done for H0:

1. `reports/audit.md`: inventory of `max/kernels` relevant to LLM decode
   (matmul/GEMV incl. quantized, attention/MHA decode, paged KV cache), with for
   each: file path, supported dtypes/quant formats, arch gating (which `sm_*` it
   compiles/dispatches for), tile/config selection mechanism, and whether it runs
   unmodified on sm_86 and sm_89.
2. `bench/results/*.json`: measurements under locked clocks for the canonical
   Llama-3-8B shapes (see `bench/shapes.py`) for:
   - MAX GEMV/matmul at M=1 and M=8, fp16 weights and every quant format MAX
     supports on GPU
   - MAX attention decode over paged KV (GQA 32/8, head_dim 128, seq 1k/4k/16k)
   - llama.cpp `mul_mat_vec_q` (Q8_0, Q4_0, Q4_K_M) and llama.cpp fattn at the
     same shapes
   - FlashInfer single-request decode at the same shapes
   - cuBLAS fp16 GEMV as the dense reference
3. `reports/h0-results.md`: tables of median µs, achieved GB/s, and % of both
   spec and measured roofline, per GPU. Every number links to a results JSON.
4. Nothing in H0 requires writing a new kernel. Do not start one.

The audit decides what H1 is. Do not presuppose the answer.

## Hard rules

- **Never invent, estimate, or extrapolate a performance number.** If something
  cannot be run, the table says `N/A` with the reason. A fabricated benchmark
  number destroys the entire purpose of this repo.
- **Every reported number comes from a results JSON produced by the harness**,
  under locked clocks, with environment recorded. See skill `bench-methodology`.
- **Correctness before performance.** No kernel or baseline gets timed until its
  output is validated against the fp32 reference with documented tolerances.
- **Your knowledge of the Mojo/MAX API is stale by default.** Mojo changes
  monthly. Treat anything you remember (keywords, import paths, `DeviceContext`
  methods, `LayoutTensor` signatures) as a hypothesis and verify it against the
  stdlib source installed in the pixi environment and the pinned `modular`
  checkout. Example: `alias` became `comptime`; `from gpu import` became
  `from std.gpu import`. Grep the installed source before writing code.
- The `modular/` checkout (git submodule, pinned SHA in `.gitmodules`) is
  **read-only**. Never edit it. Kernels we write live in `kernels/`. Upstream
  contributions are prepared in a separate clone, never here.
- Ask before: installing anything over 1 GB, downloading model weights, changing
  GPU clocks/power limits on a machine you have not confirmed is the bench box,
  or pushing to any remote.
- Write code, comments, commits, and reports in English (the audience includes
  Modular engineers). Talk to Jesús in Spanish if he writes in Spanish.
- Direct, calibrated communication. No filler, no praise, state uncertainty
  explicitly. If a result looks too good, say so and look for the measurement
  bug first.

## Repo layout

```
CLAUDE.md
.claude/skills/            project skills (read the relevant one before acting)
modular/                   git submodule, read-only, pinned SHA
bench/
  shapes.py                canonical shapes (single source of truth)
  roofline.py              spec + measured roofline computation
  env.py                   captures toolchain versions, clocks, driver, SHAs
  mojo/                    Mojo entry points that launch MAX kernels and time them
  baselines/               llama.cpp / flashinfer / cublas runners (Python/C++)
  results/                 one JSON per (impl, kernel, gpu, run); committed
kernels/                   our kernels (empty during H0)
tests/                     correctness tests (pytest for baselines, mojo test for Mojo)
reports/                   audit.md, h0-results.md, later writeups
scripts/
  gpu-lock.sh              lock/unlock clocks, persistence mode
  run-all.sh               full H0 sweep
```

## Toolchain

- Mojo/MAX via `pixi` (`pixi add modular`; check `pixi.toml`). Run Mojo with
  `pixi run mojo ...`. Verify with `pixi run mojo --version` and record it.
- Python side: `uv` for the harness env (`gguf`, `numpy`, `torch` for
  references, `flashinfer`, `pynvml`).
- CUDA toolkit + driver present on the bench box; record versions in every
  results JSON.
- Profilers: `nsys` for kernel durations (validate harness timing once per
  config), `ncu` for memory throughput when a number looks wrong.

## Canonical shapes (Llama-3-8B, decode)

hidden 4096, intermediate 14336, vocab 128256, 32 q heads, 8 kv heads,
head_dim 128. GEMV shapes (N×K): 4096×4096, 6144×4096 (fused qkv), 4096×14336,
14336×4096, 28672×4096 (fused gate/up), 128256×4096. Attention: seq_len in
{1024, 4096, 16384}, batch 1, paged KV block size as MAX defines it.
`bench/shapes.py` is the only place these live; everything imports from it.

## Working style for Claude Code

- Start every session by reading this file and the skill for the task at hand.
- Small commits, one concern each, imperative subject line.
- Before claiming a task is done, re-read the definition of done above and check
  each item literally.
- When you hit a Mojo compile error caused by API drift, fix it by reading the
  installed source, then leave a one-line note in `reports/api-drift.md`
  (old → new). This log is useful to Modular and to the forum thread.
- Keep `reports/open-questions.md` for anything that should go to
  forum.modular.com. Do not post there yourself.
