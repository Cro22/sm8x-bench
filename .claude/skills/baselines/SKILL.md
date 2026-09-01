---
name: baselines
description: How to build and run the CUDA baselines that MAX kernels are compared against — llama.cpp's mul_mat_vec_q and flash-attention kernels in isolation (not end-to-end), FlashInfer decode attention, and cuBLAS fp16 GEMV — at the canonical shapes, with the same inputs and the same timing methodology. Use whenever a task mentions llama.cpp, ggml, FlashInfer, cuBLAS, a baseline, a reference implementation, or "what does the CUDA world get on this shape".
---

# Baselines

Every MAX or in-house number is meaningless without the number a user would get
today from the tool they already run. The baselines are: llama.cpp for
quantized GEMV and attention (the local-inference incumbent), FlashInfer for
decode attention (the serving-stack incumbent), cuBLAS for dense fp16 GEMV (the
vendor ceiling). Measure the **kernel**, not the end-to-end tokens/s; tokens/s
mixes in sampling, host overhead, and scheduling that have nothing to do with
what we are studying.

All baselines: same shapes from `bench/shapes.py`, same seeded inputs from
`bench/inputs.py`, same reference from `bench/reference.py`, same statistics
and JSON schema from `bench-methodology`. Record the exact commit/version.

## llama.cpp (pin a commit in `bench/baselines/llamacpp/PIN`)

Build with CUDA for the local arch only, so the binary is not a fat binary
that muddies the picture:

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Two ways to time a single kernel, use the first if it exists at the pinned commit:

1. **`test-backend-ops` in perf mode.** `build/bin/test-backend-ops perf -o MUL_MAT`
   times individual ggml ops at built-in shapes; check whether the pinned
   version accepts custom shapes / types (`-b CUDA0`, filters). If the shapes it
   offers do not match ours, do not accept them as a substitute — go to option 2.
2. **A small C++ driver** in `bench/baselines/llamacpp/gemv_bench.cpp` linked
   against libggml: build a ggml graph with one `ggml_mul_mat(W_quant, x)` at
   our shape, run it on the CUDA backend, time with CUDA events around
   `ggml_backend_graph_compute`, K launches per sample. Same for
   `ggml_flash_attn_ext` with the KV shapes. This is ~150 lines and gives exact
   control; it is the preferred path for the final numbers.

Verify with `nsys` which CUDA kernel actually runs (`mul_mat_vec_q<...>`,
`dequantize_mul_mat_vec`, `mul_mat_q`, `flash_attn_ext_f16_...`). llama.cpp
picks different kernels by shape, batch, and arch; the JSON `variant` field
must name the kernel nsys shows, not the ggml op.

Also record llama.cpp's own numbers for context: `llama-bench` on a real model
(`-p 0 -n 128`) gives tokens/s. Report it as context only, never in the same
table as kernel-level numbers.

## FlashInfer

`uv add flashinfer` (match the CUDA/torch versions; the wheel index is version
specific — read their README for the current install line, do not guess).

`flashinfer.single_decode_with_kv_cache(q, k, v)` for single sequence; for the
paged variant use `BatchDecodeWithPagedKVCacheWrapper` with batch 1 and the page
size MAX uses, so the paged-read pattern is comparable. Time with
`torch.cuda.Event`, K launches per sample, `torch.cuda.synchronize()` around
the batch. GQA: q heads 32, kv heads 8, head_dim 128, fp16 KV. Sequence lengths
from `shapes.py`.

Check with nsys that the timed region contains exactly one kernel per call (the
wrapper's `plan()` must be outside the timed loop).

## cuBLAS fp16 GEMV

Via `torch`: `torch.matmul(W_fp16, x_fp16)` with W as N×K fp16 and x as K×1
(and K×8 for M=8), fp32 accumulation (`torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False`).
Time with CUDA events. Confirm with nsys that it dispatches a gemv/gemvx
kernel and not a GEMM tile for M=1. If it does dispatch GEMM, report what it
did. This is the dense ceiling: 2 bytes/weight, so % of roofline is directly
comparable across fp16 implementations.

## PyTorch reference (correctness only)

`bench/reference.py` computes the fp32 reference on CPU (numpy) — not on the
GPU — so the reference is independent of any CUDA math library. It is slow for
the vocab GEMV (128256×4096 fp32 ≈ 2 GB); cache references to
`bench/reference_cache/*.npy` keyed by (shape, format, seed).

## Reporting asymmetries honestly

- llama.cpp fuses dequant; if MAX dequantizes to fp16 first, both get measured
  as they are, and the table notes the extra traffic (see `gguf-quant-formats`).
- llama.cpp's attention may use fp16 accumulation on some paths; note it.
- FlashInfer is tuned for datacenter too; if it underperforms on the 4090 that
  is itself a finding, not a reason to drop it.
- If a baseline cannot be built at the pinned commit on this box, the table
  says N/A with the error, and `reports/open-questions.md` gets a line.
