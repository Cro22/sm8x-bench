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

- **Dense GEMV (fp16/bf16)** — 87–98 % of the memory-bandwidth roofline at M=1;
  no gap to close (and it beats cuBLAS at M=1, which falls to a GEMM tile).
  ([details](reports/h0-results.md))
- **Attention decode** (flash-decoding over paged KV) — at the roofline at long
  context (**91.0 % at 16k, a hair ahead of FlashInfer**) with a narrow mid-context
  gap: at seq 4096 MAX is 63.7 % vs FlashInfer's 70.2 % on the same paged KV
  (~6–7 pts, same-session 3 passes each, non-overlapping). A tuning gap at seq~4k,
  not "nothing to improve" — details in [reports/h0-results.md](reports/h0-results.md).
- **Q4_0 (4-bit) matmul, MAX's real dispatch** — uneven: it **has a decode-tuned
  config** for up_proj/down_proj (69–81 % of roofline), **falls to a datacenter
  GEMM tile** for gate_up/lm_head (15 %), and **fails to compile** for
  GGUF Q4_0 (group_size 32) on o_proj/qkv. MAX has no single decode-quant kernel
  that works well everywhere.
- **Our Q4_0 GEMV** ([`kernels/q4_0_gemv.mojo`](kernels/q4_0_gemv.mojo)) —
  **74–102 % of roofline on every shape, at parity with llama.cpp, and faster than
  MAX on all six.** After tuning (multiple output rows per warp for memory-level
  parallelism + a per-shape launch config) it **matches** llama.cpp Q4_0 —
  measured same-session, 3 passes each: five of six shapes are ties within the
  0–9 % run-to-run noise (the locked graphics clock doesn't lock the GDDR6X memory
  clock and the shared desktop steals bandwidth), and llama.cpp is faster on
  gate_up. No shape is a robust win for ours. The **real** win is vs MAX: ~6×
  faster on gate_up/lm_head, ahead on up_proj/down_proj, and it runs the two shapes
  MAX compile-fails. A single kernel, uniformly near-roofline, where MAX is absent,
  15 %, or (at best) 81 %.

### Q4_0 decode GEMV, three implementations, same weights, same measurement

nsys per-kernel time / % of the 936 GB/s spec roofline, M=1, RTX 3090, graphics
clock verified 1695 MHz per run (memory clock NOT lockable on GeForce — see below).
MAX measured through its **real** public dispatcher (`matmul_gpu_qint4[g32]`),
llama.cpp with CUDA graphs disabled. **ours and llama.cpp: same-session, 3 passes
each** (median / least-contended min shown). This summary is a hand-transcribed
rounded copy of the JSON-generated tables in
[reports/h0-results.md](reports/h0-results.md).

| Llama-3-8B shape (N×K) | MAX (real dispatch) | **ours** med/min% | llama.cpp med/min% |
|---|---|---|---|
| o_proj 4096×4096       | compile-fail (g32) | **13.7 µs / 73.8 / 74.5** | 13.9 µs / 72.6 / 72.7 |
| qkv 6144×4096          | compile-fail (g32) | **19.6 µs / 77.4 / 82.5** | 18.1 µs / 83.7 / 83.8 |
| down_proj 4096×14336   | 43.5 µs / 81.2 %   | **38.4 µs / 92.0 / 92.0** | 38.0 µs / 92.9 / 94.7 |
| up_proj 14336×4096     | 51.6 µs / 68.4 %   | **41.7 µs / 84.8 / 88.1** | 39.9 µs / 88.5 / 88.5 |
| gate_up 28672×4096     | 474 µs / 14.9 %    | **77.0 µs / 91.7 / 91.8** | 70.6 µs / 100.1 / 100.1 |
| lm_head 128256×4096    | 2031 µs / 15.6 %   | **312 µs / 101.2 / 101.9** | 311 µs / 101.5 / 101.5 |

Ours is at **parity** with llama.cpp Q4_0 (five of six shapes tie within the 0–9 %
run-to-run band; llama.cpp is faster on gate_up), faster than MAX on the four
shapes MAX can run, and it runs the two MAX compile-fails (no MAX speed to beat
there). The band is wide because a locked graphics clock does not lock the GDDR6X memory
clock and the desktop compositor steals bandwidth — so per-shape gaps under ~5 %
are noise. Full tables (dense fp16/bf16, Q8_0/Q4_K, attention, cuBLAS, FlashInfer,
the bandwidth probe) with links to every results JSON:
**[reports/h0-results.md](reports/h0-results.md)**.

---

## Why MAX's Q4_0 is uneven, and what our kernel does

MAX's int4 path is a **tensor-core GEMM**. Where the dispatch has a decode-tuned
config (small BM), it does well (down_proj/up_proj, 69–81 %). Where a shape falls
to the default 128×128 tile (gate_up/lm_head), at M=1 it uses 1 of 128 tile rows
(1/128 of the tensor-core work and A-bandwidth wasted) — hence 15 %; the limiter
is *inferred* to be per-weight dequant compute, not measured with ncu (blocked on
this box). And for group_size 32 with a BK=128 tuned config
(o_proj/qkv) it **fails to compile**. So the slowness is a *design bias*
(throughput/datacenter tiling with no uniform decode-quant kernel), while the
compile failure is a *bug*. Full design-vs-bug analysis with file:line:
**[reports/max-q4_0-analysis.md](reports/max-q4_0-analysis.md)**.

The compile failure is reproduced through the **public** dispatcher
(`matmul_gpu_qint4[group_size=32]`, not a forced config): `32 // BK(128) == 0`
→ an `int_tuple` out-of-bounds at `qmatmul_gpu.mojo:873`. It is a genuine MAX
limitation, not a harness setup error — the *identical* harness at the same
`group_size=32` compiles and runs validated for up_proj/down_proj (BK=32 configs)
and fails only on o_proj/qkv. The full compiler output is committed alongside each
compile-fail JSON as `bench/results/max_gemv_Q4-0-M1-{o_proj,qkv_fused}.compile_error.txt`.

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
  the *identical* GGUF Q4_0 bytes and the same seeded activation values — though
  llama.cpp consumes them as **F32** while MAX and ours use **BF16** (a noted
  asymmetry; the weight traffic that dominates is identical). The **ours and
  llama.cpp Q4_0** JSONs record the sha256 of the exact W/x/ref bytes read; the
  older MAX/attention/cuBLAS JSONs predate that harness change and do **not**
  carry input hashes (their inputs are the same committed generators).
- **No number is hand-typed in the generated tables.** Every cell in
  [reports/h0-results.md](reports/h0-results.md) comes from a committed results
  JSON via `bench/report.py`; a value that can't be measured is `N/A` with the
  reason. (The condensed summary table above is a hand-transcribed rounded copy of
  those.)

## Reproduce

```bash
# toolchain (uv): max==26.5.0 / Mojo 1.0.0; modular submodule at tag max/v26.5.0
uv sync && git submodule update --init --depth 1
# build the baseline drivers once (llama.cpp src/ must be built with CUDA sm_86 first —
# cmake -B bench/baselines/llamacpp/src/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
# then cmake --build that dir; the driver build steps are committed scripts):
bash bench/baselines/llamacpp/build_gemv_bench.sh    # gemv_bench + quantize_weight
bash bench/baselines/cublas/build_cublas_gemv.sh     # cuBLAS fp16 driver

# lock the GRAPHICS clock on the Windows host (admin): nvidia-smi -lgc 1695,1695
# (the GDDR6X memory clock is NOT lockable on GeForce; it and desktop contention give
#  a 0-9% run-to-run band — the graphics clock is sampled under load and recorded per JSON)
uv run python -m bench.run_bw_probe --locked-clock 1695              # measured roofline
scripts/sweep_gemv.sh 816.3 1695                                     # MAX dense fp16/bf16
uv run python -m bench.run_gemv_max_dispatch --shape gate_up --M 1 --locked-clock 1695 --measured-gbps 816.3   # MAX Q4_0 REAL dispatch
uv run python -m bench.run_gemv_ours      --shape o_proj --M 1 --locked-clock 1695 --measured-gbps 816.3       # our kernel (3 passes)
uv run python -m bench.run_gemv_llamacpp  --shape o_proj --M 1 --locked-clock 1695 --measured-gbps 816.3       # baseline (3 passes)
uv run python -m bench.report                                        # regenerate tables
# full sweep incl. Q8_0/Q4_K/cuBLAS/FlashInfer: see reports/h0-results.md "Reproduce"
```

## Status

- **H0 (audit + benchmark upstream MAX):** the audit and the RTX 3090
  measurements are done. Dense + attention are at the roofline. Q4_0 is uneven
  (see the table): MAX has a good decode config for some shapes, a datacenter
  GEMM (15 %) for others, and a g32 compile failure for two. llama.cpp GEMV
  baselines cover **Q4_0, Q8_0 and Q4_K** (Q8_0/Q4_K are bandwidth-saturated and
  have no MAX GPU counterpart — CPU-only upstream). The **cuBLAS fp16 GEMV** dense
  ceiling is measured too — and at M=1 cuBLAS falls to a tensor-core GEMM tile on
  five of six shapes, so **MAX's split-K fp16 GEMV actually beats cuBLAS** there
  (e.g. o_proj 90.6 % vs 79.2 %). The **FlashInfer decode-attention** baseline
  (paged, page 128, same-session 3-pass) surfaces a real finding: MAX is at the
  roofline and slightly ahead at long context (91.0 % vs 89.9 % at seq 16384) but
  trails ~6–7 pts at seq 4096 (**63.7 % vs 70.2 %**, non-overlapping distributions).
  A narrow, confirmed mid-context tuning gap. Only llama.cpp flash-attn remains.
- **H1 (write the gap kernel):** our Q4_0 GEMV is 74–102 % of roofline on all six
  shapes — at **parity** with llama.cpp (same-session 3-pass: five ties, llama.cpp
  faster on gate_up; no robust ours win) — and faster than MAX on the four shapes
  MAX can run (~6× on gate_up/lm_head); it also runs the two where MAX compile-fails
  (no MAX speed to beat there). Uniform coverage at the baseline's level, not
  beating it. M=1 only.
- **Not yet:** M>1 for our kernel, the residual quantize-launch overhead on the
  smallest shapes, sm_89 (no 4090 on the bench box), the last baseline (llama.cpp
  flash-attn), raw `.nsys-rep` artifacts, and preparing the upstream contribution.

This record was corrected after **two** adversarial reviews. The first found MAX
had been measured through a forced fallback config (not its real dispatcher) with
unproven clocks. The second found the report still oversold rigor — so **"beats
llama.cpp on four of six" is retracted** (same-session 3-pass measurement shows
parity), input hashes and all 3 pass medians are now stored in each JSON,
"compute-bound" is relabeled an inferred diagnosis, and the memory clock (which
these bandwidth-bound kernels actually depend on) is documented as unlockable and
uncontrolled. See reports/h0-results.md "Corrections".

All results are from the RTX 3090; the RTX 4090 (sm_89) is not present on the
current bench box, so those rows are `N/A` for now.
