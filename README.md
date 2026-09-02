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

On consumer Ampere, MAX's **dense** decode kernels are already at the roofline;
its **4-bit** path is uneven — good on some shapes, absent or datacenter-slow on
others. We wrote a Q4_0 decode GEMV that runs well and uniformly across all
shapes, close to llama.cpp.

- **Dense GEMV (fp16/bf16)** — 87–98 % of the memory-bandwidth roofline at M=1.
  Nothing to improve. ([details](reports/h0-results.md))
- **Attention decode** (flash-decoding over paged KV) — near the roofline at long
  context (86 % of spec at 16k). Nothing to improve.
- **Q4_0 (4-bit) matmul, MAX's real dispatch** — uneven: it **has a decode-tuned
  config** for up_proj/down_proj (69–81 % of roofline), **falls to a datacenter
  GEMM tile** for gate_up/lm_head (15 %), and **fails to compile** for
  GGUF Q4_0 (group_size 32) on o_proj/qkv. MAX has no single decode-quant kernel
  that works well everywhere.
- **Our Q4_0 GEMV** ([`kernels/q4_0_gemv.mojo`](kernels/q4_0_gemv.mojo)) —
  **61–92 % of roofline on every shape, ~5–15 % behind llama.cpp, and it runs
  even where MAX compile-fails.** It is ~6× faster than MAX only where MAX uses
  the datacenter GEMM (gate_up/lm_head); MAX's decode config is competitive or
  faster on down_proj. Not categorically faster than MAX — the value is a single
  kernel that is uniformly near-roofline.

### Q4_0 decode GEMV, three implementations, same weights, same measurement

nsys per-kernel time / % of the 936 GB/s spec roofline, M=1, RTX 3090, graphics
clock verified at 1695 MHz per run; MAX measured through its **real** public
dispatcher (`matmul_gpu_qint4[g32]`), llama.cpp with CUDA graphs disabled so each
launch is a distinct kernel instance. Numbers are `bench/report.py`-generated
from the results JSON (see the linked tables); this summary is rounded.

| Llama-3-8B shape (N×K) | MAX (real dispatch) | **ours** | llama.cpp |
|---|---|---|---|
| o_proj 4096×4096       | compile-fail (g32) | **16.6 µs / 61 %** | 14.3 µs / 71 % |
| qkv 6144×4096          | compile-fail (g32) | **21.8 µs / 69 %** | 19.5 µs / 78 % |
| down_proj 4096×14336   | 43.6 µs / 81 %     | **50.1 µs / 70 %** | 40.3 µs / 88 % |
| up_proj 14336×4096     | 51.4 µs / 69 %     | **44.1 µs / 80 %** | 41.1 µs / 86 % |
| gate_up 28672×4096     | 474 µs / 15 %      | **80.6 µs / 88 %** | 76.5 µs / 92 % |
| lm_head 128256×4096    | 2057 µs / 15 %     | **344 µs / 92 %**  | 329 µs / 96 % |

llama.cpp is the fastest on every shape; ours is 5–15 % behind it; MAX is
best-in-class on down_proj but compile-fails or drops to 15 % elsewhere. Full
tables (dense fp16/bf16, attention, the bandwidth probe) with links to every
results JSON: **[reports/h0-results.md](reports/h0-results.md)**.

---

## Why MAX's Q4_0 is uneven, and what our kernel does

MAX's int4 path is a **tensor-core GEMM**. Where the dispatch has a decode-tuned
config (small BM), it does well (down_proj/up_proj, 69–81 %). Where a shape falls
to the default 128×128 tile (gate_up/lm_head), at M=1 it uses 1 of 128 tile rows
and is dominated by per-weight int→float dequantization — **compute-bound, not
memory-bound** — hence 15 %. And for group_size 32 with a BK=128 tuned config
(o_proj/qkv) it **fails to compile**. So the slowness is a *design bias*
(throughput/datacenter tiling with no uniform decode-quant kernel), while the
compile failure is a *bug*. Full design-vs-bug analysis with file:line:
**[reports/max-q4_0-analysis.md](reports/max-q4_0-analysis.md)**.

Our kernel does what llama.cpp does: quantize the activations to **Q8_1** once,
then compute the Q4_0·Q8_1 dot with **`dp4a` int8 multiply-accumulate** (emitted
as inline PTX — the Mojo stdlib has no `dp4a` on sm_86), unpacking nibbles with a
`0x0F0F0F0F` mask. That makes it memory-bound and uniformly near-roofline on all
six shapes, including the ones where MAX compile-fails or drops to 15 %.

Upstream questions (a compile-fix for g32, a latent out-of-bounds A-tile load
that only manifests when BM > M and K is large) are drafted in
**[reports/open-questions.md](reports/open-questions.md)** — the latter is a
latent bug our forced-config experiment exposed, not something MAX's real
dispatch hits at these shapes.

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
# build the llama.cpp baseline driver once (needs the built libggml; see bench/baselines/llamacpp/):
#   cmake -B bench/baselines/llamacpp/src/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 ... && cmake --build ...
#   g++ ... bench/baselines/llamacpp/gemv_bench.cpp -lggml ... -o bench/baselines/llamacpp/gemv_bench

# lock clocks on the Windows host (admin): nvidia-smi -lgc 1695,1695  (verified per-run in each JSON)
uv run python -m bench.run_bw_probe --locked-clock 1695              # measured roofline
scripts/sweep_gemv.sh 816.3 1695                                     # MAX dense fp16/bf16
uv run python -m bench.run_gemv_max_dispatch --shape gate_up --M 1 --locked-clock 1695 --measured-gbps 816.3   # MAX Q4_0 REAL dispatch
uv run python -m bench.run_gemv_ours      --shape o_proj --M 1 --locked-clock 1695 --measured-gbps 816.3       # our kernel
uv run python -m bench.run_gemv_llamacpp  --shape o_proj --M 1 --locked-clock 1695 --measured-gbps 816.3       # baseline
uv run python -m bench.report                                        # regenerate tables
```

## Status

- **H0 (audit + benchmark upstream MAX):** the audit and the RTX 3090
  measurements are done. Dense + attention are at the roofline. Q4_0 is uneven
  (see the table): MAX has a good decode config for some shapes, a datacenter
  GEMM (15 %) for others, and a g32 compile failure for two. Secondary baselines
  (Q8_0/Q4_K, FlashInfer, cuBLAS) are not done yet — so "best CUDA baselines" so
  far means llama.cpp Q4_0 only.
- **H1 (write the gap kernel):** our Q4_0 GEMV is 61–92 % of roofline on all six
  shapes, ~5–15 % behind llama.cpp, and runs where MAX compile-fails or drops to
  15 %. It does not beat MAX everywhere (MAX's decode config wins on down_proj).
  M=1 only.
- **Not yet:** M>1 for our kernel, closing the last ~10–15 % to llama.cpp,
  sm_89 (no 4090 on the bench box), the secondary baselines, and preparing the
  upstream contribution.

This record was corrected after an adversarial review found the first pass had
measured MAX through a forced fallback config (not its real dispatcher) and had
unproven clocks; those are fixed here (real dispatch, per-run clock verification,
graphs-disabled llama.cpp).

All results are from the RTX 3090; the RTX 4090 (sm_89) is not present on the
current bench box, so those rows are `N/A` for now.
