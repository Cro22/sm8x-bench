# sm8x-bench

A rigorous benchmark workbench for LLM inference kernels on **consumer** NVIDIA
GPUs (RTX 3090 = sm_86, RTX 4090 = sm_89). It measures the kernels that ship in
Modular's open-source **MAX** engine (written in Mojo) against the best CUDA
baselines (llama.cpp, cuBLAS, FlashInfer) and against the hardware roofline — on
hardware Modular does not tune for — and, where the audit finds a real gap,
writes a better kernel intended for upstreaming to `modular/max/kernels`.

This is a workbench and a public record, **not** a competing kernel library.

Owner: Jesús ([Cro22](https://github.com/Cro22)). Methodology reused from
[mojo-cuda-ampere](https://github.com/Cro22/mojo-cuda-ampere).

---

## Headline result (RTX 3090, sm_86)

On consumer Ampere, MAX's **dense** decode kernels are already excellent, and its
**4-bit** path was not — so we fixed the 4-bit path:

- **Dense GEMV (fp16/bf16)** — 87–98 % of the memory-bandwidth roofline at M=1.
  Nothing to improve. ([details](reports/h0-results.md))
- **Attention decode** (flash-decoding over paged KV) — 86 % of spec / 99 % of
  the measured roofline at 16k context. Nothing to improve.
- **Q4_0 (4-bit) matmul** — MAX runs at **6–15 % of roofline**, ~6–10× slower
  than llama.cpp, its tuned int4 configs **don't compile** for GGUF Q4_0, and one
  Llama-3 shape **crashes**. This was the gap.
- **Our Q4_0 GEMV** ([`kernels/q4_0_gemv.mojo`](kernels/q4_0_gemv.mojo)) —
  **62–93 % of roofline, at parity with llama.cpp (beats it on one shape),
  6–10× faster than MAX**, and runs the shape MAX crashes on.

### Q4_0 decode GEMV, three implementations, same weights, same measurement

nsys per-kernel time / % of the 936 GB/s spec roofline, M=1, RTX 3090 @ 1695 MHz:

| Llama-3-8B shape (N×K) | MAX (upstream) | **ours** | llama.cpp |
|---|---|---|---|
| o_proj 4096×4096       | 166 µs / 6 %   | **16.2 µs / 62 %** | 16 µs / 62 % |
| qkv 6144×4096          | 164 µs / 9 %   | **21.7 µs / 70 %** | 23 µs / 66 % |
| down_proj 4096×14336   | **crash**      | **48.8 µs / 72 %** | 45 µs / 78 % |
| up_proj 14336×4096     | 325 µs / 11 %  | **44.8 µs / 79 %** | 45 µs / 79 % |
| gate_up 28672×4096     | 487 µs / 14 %  | **81.9 µs / 86 %** | 77 µs / 91 % |
| lm_head 128256×4096    | 2094 µs / 15 % | **341 µs / 93 %**  | 330 µs / 96 % |

Full tables (dense fp16/bf16, attention, the bandwidth probe) with links to every
results JSON: **[reports/h0-results.md](reports/h0-results.md)**.

---

## Why MAX's Q4_0 was slow, and what we changed

MAX launches a **tensor-core int4 GEMM** (128×128 tile) for Q4_0. That is a
throughput design: at M=1 decode it uses 1 of 128 tile rows and is dominated by
per-weight int→float dequantization — **compute-bound, not memory-bound**. Our
kernel does what llama.cpp does: quantize the activations to **Q8_1** once, then
compute the Q4_0·Q8_1 dot with **`dp4a` int8 multiply-accumulate** (emitted as
inline PTX — the Mojo stdlib has no `dp4a` on sm_86), unpacking nibbles with a
`0x0F0F0F0F` mask. That makes the kernel memory-bound and it reaches the roofline.

Whether MAX's slowness is a deliberate datacenter-throughput choice versus a
defect — and the root cause of the compile failure and the crash — is analyzed
in **[reports/max-q4_0-analysis.md](reports/max-q4_0-analysis.md)**, with the
specific upstream questions in
**[reports/open-questions.md](reports/open-questions.md)**.

---

## What's in here (documents)

| Document | What it is |
|---|---|
| [reports/audit.md](reports/audit.md) | Inventory of `max/kernels` for LLM decode: what exists, arch gating (sm_86/89), config selection, what runs vs crashes, ranked gaps. Every claim cites `file:line`. |
| [reports/h0-results.md](reports/h0-results.md) | The measurements: dense GEMV, attention decode, Q4_0 (MAX / ours / llama.cpp), roofline. Tables generated from JSON by `bench/report.py`; each row links its results JSON. |
| [reports/max-q4_0-analysis.md](reports/max-q4_0-analysis.md) | Deep read of MAX's Q4_0 path: design-vs-bug verdict per issue (slowness, compile failure, crash). |
| [reports/open-questions.md](reports/open-questions.md) | Reproducible questions/bugs drafted for the Modular forum (never auto-posted). |
| [reports/api-drift.md](reports/api-drift.md) | Mojo 1.0.0 / MAX 26.5.0 API changes we hit (e.g. `fn` removed → `def`), a byproduct useful to Modular. |
| [CLAUDE.md](CLAUDE.md) | Project rules, phases, and hard constraints. |

## Repo layout

```
kernels/            our kernels (q4_0_gemv.mojo) — upstream candidates
bench/
  shapes.py         canonical Llama-3-8B shapes (single source of truth)
  roofline.py       spec + measured roofline, minimum-traffic byte counts
  env.py            captures GPU state + toolchain/SHAs into every results JSON
  nsys.py           run under nsys, parse per-kernel duration (authoritative timing)
  reference.py      seeded inputs + fp32 reference (fp16/bf16/Q4_0) as raw .bin
  results_io.py     the one JSON writer (schema in the bench-methodology skill)
  report.py         results JSON -> Markdown tables
  run_*.py          runners: MAX (gemv/attention), llama.cpp, ours
  mojo/             Mojo entry points that launch the kernels and time them
  baselines/        llama.cpp (pinned) + its Q4_0 gemv_bench.cpp driver
  results/          one JSON per (impl, kernel, shape); committed
modular/            git submodule, read-only, pinned to tag max/v26.5.0
reports/            audit, results, analysis, open questions, api drift
scripts/            gpu-lock.sh, sweep_gemv.sh
```

## Methodology (why the numbers are trustworthy)

- **Locked clocks.** Graphics clock locked to 1695 MHz (3090) for every run;
  memory clock isn't lockable on GeForce. See `scripts/gpu-lock.sh`.
- **Authoritative timing is nsys per-kernel duration**, not wall-clock — it
  excludes host-side dispatch overhead (amortized in production) and is robust to
  the desktop sharing the GPU. Cross-checked that the harness agrees where the
  dispatch is cheap.
- **Correctness before timing.** Every kernel is validated against an fp32
  reference (relative L2 error — the standard GEMM metric, robust to
  cancellation) before any number is recorded.
- **Same weights across implementations.** MAX, llama.cpp, and our kernel read
  the *identical* GGUF Q4_0 bytes and the same seeded activations.
- **No number is hand-typed.** Every table cell comes from a committed results
  JSON via `bench/report.py`; a value that can't be measured is `N/A` with the
  reason.

## Reproduce

```bash
# toolchain (uv): max==26.5.0 / Mojo 1.0.0; modular submodule at tag max/v26.5.0
uv sync && git submodule update --init --depth 1

# lock clocks on the Windows host (admin): nvidia-smi -lgc 1695,1695
uv run python -m bench.run_bw_probe --locked-clock 1695              # measured roofline
scripts/sweep_gemv.sh 815.5 1695                                     # MAX dense fp16/bf16
uv run python -m bench.run_gemv_max   --shape o_proj --fmt Q4_0 --M 1 --locked-clock 1695 --measured-gbps 815.5   # MAX Q4_0
uv run python -m bench.run_gemv_ours  --shape o_proj --M 1 --locked-clock 1695 --measured-gbps 815.5              # our kernel
uv run python -m bench.run_gemv_llamacpp --shape o_proj --M 1 --locked-clock 1695 --measured-gbps 815.5          # baseline
uv run python -m bench.report                                        # regenerate tables
```

## Status

- **H0 (audit + benchmark upstream MAX):** done on the RTX 3090. Dense + attention
  are at the roofline; Q4_0 is the gap (3.5–9.7× behind llama.cpp, plus a compile
  failure and a crash).
- **H1 (write the gap kernel):** our Q4_0 GEMV reaches llama.cpp parity and is
  6–10× over MAX, for M=1.
- **Not yet:** M>1 for our kernel, sm_89 (no 4090 on the bench box yet), the
  secondary baselines (Q8_0/Q4_K, FlashInfer, cuBLAS), and preparing the upstream
  contribution.

All results are from the RTX 3090; the RTX 4090 (sm_89) is not present on the
current bench box, so those rows are `N/A` for now.
