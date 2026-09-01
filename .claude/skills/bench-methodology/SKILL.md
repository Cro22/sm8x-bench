---
name: bench-methodology
description: How every GPU kernel measurement in this repo is taken and recorded — clock locking, warmup, 30-sample median with IQR, spec and measured roofline, results JSON schema, and cross-checking with nsys. Use this skill whenever you time a kernel, add a benchmark entry point, write a results file, compute achieved bandwidth or % of roofline, or compare implementations. Also use it when a number looks surprisingly good or bad, since the first suspect is always the measurement.
---

# Benchmark methodology

The credibility of this repo rests on numbers that a Modular kernel engineer
would accept without asking follow-up questions. That means: fixed clocks,
robust statistics, roofline context, full provenance, and no unexplained
outliers. Follow the same procedure for MAX kernels, CUDA baselines, and our own
kernels, so comparisons are apples to apples.

## Procedure per measurement

1. **Confirm the machine is in bench state.** `scripts/gpu-lock.sh status` must
   show persistence mode on and graphics clock locked. If not, run
   `scripts/gpu-lock.sh lock` (needs sudo; ask Jesús if it fails). Memory clock
   locking (`-lmc`) is usually rejected on GeForce; record that it was not
   locked. Record the locked clock value — the compute roofline depends on it.
2. **Confirm nothing else is on the GPU.** `nvidia-smi --query-compute-apps=pid
   --format=csv` should list only your process. A desktop session compositing
   on the same GPU adds noise; note it if unavoidable.
3. **Validate correctness first** (see `mojo-gpu-kernel` and `baselines`). Never
   time an unvalidated kernel.
4. **Warmup:** at least 10 untimed launches, or until two consecutive
   measurements are within 2%.
5. **Sample:** 30 timed measurements. Each measurement times a batch of K
   back-to-back launches between two device syncs and divides by K, where K is
   chosen so one batch takes ≥ 5 ms (avoids launch-overhead and timer
   resolution dominating small GEMVs). Record K.
6. **Statistics:** report median, IQR (Q1, Q3), min, and p95. The median is the
   headline number. If IQR/median > 5%, something is wrong (clocks, thermal,
   contention) — investigate before recording.
7. **Cross-check** once per (kernel, shape) with `nsys profile --stats=true`:
   the kernel's mean duration should agree with the harness median within ~5%.
   If not, the harness is measuring something other than the kernel (H2D copies,
   allocation, host overhead). Fix it. Record the nsys number in the JSON.
8. **Write the results JSON** (schema below) via `bench/results_io.py`. One
   file per (impl, kernel, gpu, run). Commit it.

## Timing in Mojo

Do not trust memory about `DeviceContext` timing APIs. Before writing the timer,
grep the installed stdlib for event/elapsed helpers:

```
grep -rn "elapsed\|Event\|timing" $(pixi run python -c "import sys") ...
grep -rn "elapsed\|class Event\|struct .*Event" modular/mojo/stdlib/std/gpu/host/
```

Prefer a device-event-based timer if one exists. Otherwise: `ctx.synchronize()`,
`time.perf_counter_ns()`, K launches, `ctx.synchronize()`, `perf_counter_ns()`.
Either way, step 7 (nsys cross-check) is mandatory for the first run of each
config, precisely because the host-side timer includes sync latency.

## Roofline

Two roofline numbers are reported, always both:

- **Spec roofline:** from `references/hardware.md` (bandwidth from memory clock
  × bus width; FP32 FLOPS = SMs × 128 FMA × 2 × *locked* graphics clock).
- **Measured roofline:** bandwidth from `bench/mojo/bw_probe.mojo`, a read-only
  streaming reduction over ≥ 2 GB, best of 30, run in the same session under the
  same clocks. Consumer cards typically reach 88–93% of spec. Percent of
  measured roofline is the fairer number; percent of spec is what people expect
  to see. Report both.

Bytes moved for a kernel = the minimum traffic implied by the algorithm, not
what the kernel happens to do:

- GEMV N×K, format F: `N*K*bytes_per_element(F) + K*2 (fp16 x) + N*2 (fp16 y)`.
  For quant formats use the exact block byte size (Q8_0 = 34 B/32, Q4_0 =
  18 B/32, Q4_K = 144 B/256).
- Decode attention: `seq_len * kv_heads * head_dim * 2 (K) * 2 (K and V) *
  bytes(kv dtype)` plus negligible Q/O.

`bench/roofline.py` implements these; shapes come from `bench/shapes.py`. Do
not compute bytes by hand in a report.

## Results JSON schema

```json
{
  "schema": 1,
  "impl": "max|llamacpp|flashinfer|cublas|ours",
  "kernel": "gemv|attention_decode|matmul|bw_probe",
  "variant": "Q4_0 | fp16 | mul_mat_vec_q ...",
  "gpu": {"name": "RTX 3090", "arch": "sm_86", "sms": 82, "driver": "...",
          "graphics_clock_mhz_locked": 1695, "mem_clock_mhz": 9751,
          "mem_clock_locked": false, "power_limit_w": 350},
  "shape": {"M": 1, "N": 4096, "K": 4096},
  "dtype": {"weights": "Q4_0", "activations": "fp16", "accum": "fp32"},
  "bytes_moved": 0, "flops": 0,
  "timing": {"n_samples": 30, "launches_per_sample": 64,
             "median_us": 0, "q1_us": 0, "q3_us": 0, "min_us": 0, "p95_us": 0,
             "nsys_mean_us": 0},
  "achieved_gbps": 0, "achieved_tflops": 0,
  "roofline": {"spec_gbps": 936, "measured_gbps": 0,
               "pct_spec": 0, "pct_measured": 0},
  "correctness": {"validated": true, "max_abs_err": 0, "max_rel_err": 0,
                  "tolerance": "rtol=1e-2 atol=1e-3"},
  "env": {"mojo_version": "", "modular_sha": "", "cuda_version": "",
          "llamacpp_sha": "", "flashinfer_version": "", "harness_sha": "",
          "timestamp_utc": ""},
  "notes": ""
}
```

`bench/env.py` fills `gpu` and `env` automatically (pynvml + `git rev-parse`).
Never hand-edit a results file.

## Comparing implementations

- Same shapes, same session, same clocks, same input data (seeded).
- A comparison table always includes the roofline column; "A is 1.3× faster
  than B" without "A is at 62% of roofline" is not informative.
- When a baseline supports a fused variant (e.g., llama.cpp fuses dequant into
  the dot product) and MAX does not, benchmark what each actually does and note
  the asymmetry. Do not handicap either side to make them "equal".
- Consumer cards throttle under sustained load even with locked clocks if the
  power limit is hit. For the large shapes (vocab GEMV, 16k attention) watch
  `nvidia-smi -q -d PERFORMANCE` for throttle reasons during the run and record
  any.

## When a number looks wrong

In this order: (1) clocks actually locked? (2) correctness test actually passed
on this exact config? (3) nsys agrees with harness? (4) bytes_moved computed for
the right format? (5) L2 effects — the 4090 has 72 MB of L2, so a 4096×4096 Q4_0
weight (9 MB) is L2-resident on repeated launches and will show >100% of DRAM
roofline. That is a real effect, not a bug, but it must be called out and the
sweep must include shapes that exceed L2 (vocab GEMV, 4096×14336 fp16). For the
3090 (6 MB L2) this matters only for the smallest shapes.

Read `references/hardware.md` for the per-GPU constants.
