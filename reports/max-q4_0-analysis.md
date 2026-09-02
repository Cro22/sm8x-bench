# MAX GPU Q4_0 (int4) matmul on consumer Ampere: design bias vs. defects

Scope: `modular/max/kernels/src/quantization/qmatmul_gpu.mojo` and its callers.
Hardware: RTX 3090 (sm_86). MAX 26.5.0. All runs use `bench/mojo/qgemv_max`,
which repacks Q4_0 once and calls MAX's `multistage_gemm_q` directly.

TL;DR: MAX's GPU int4 path is a **software-pipelined tensor-core GEMM designed
for throughput (large M)**. It has **no GEMV/decode-specialized quant kernel** on
NVIDIA. On top of that structural bias sit two independent **defects** that hurt
consumer decode specifically: a dispatch that crashes at compile time for
group_size=32 (A), and an unmasked A-tile load that reads out of bounds when
M < BM and faults for large K (B).

---

## 1. Is the Q4_0 GPU kernel a throughput (GEMM) or a decode (GEMV) design?

It is a **tensor-core GEMM**, not a GEMV.

- The MMA loop issues real `mma.sync` tensor-core instructions:
  `multistage_mma_q` builds `TensorCore[accum, a_type, mma_shape, transpose_b]`
  (qmatmul_gpu.mojo:485) and calls `mma_op.mma(...)` in the K loop
  (qmatmul_gpu.mojo:552). For bf16 in / fp32 accum on NVIDIA the shape is
  `16x8x16` (tensor_core.mojo:1395-1397). The int4 weights are dequantized to
  bf16 in registers and fed straight into the tensor cores
  (tensor_core.mojo:1200 `int4tobf16`, called from `load_b`).
- It is tiled for large M. Blocking comes from `MatmulConfig.block_tile_shape`
  `[BM, BN, BK]`; the grid is `ceildiv(N,BN) x ceildiv(M,BM) x num_k_partitions`
  (utils_gpu.mojo:242-247). The A tile loaded each K-step is a full `[BM, BK]`
  block (qmatmul_gpu.mojo:468, 507, 823).
- Quantify M=1 waste with the only Q4_0-compatible config, `default_config`
  (block 128x128x32, warp 64x64x32; qmatmul_gpu.mojo:2341-2347): BM=128, so
  `grid.y = ceildiv(1,128) = 1` block computes a 128-row output tile with **one
  valid row → 1/128 = 0.78% M-tile utilization**. The block runs 4 warps
  (2x2), each doing WM=64 rows of MMAs; ~127/128 of the tensor-core work and of
  the A-tile bandwidth is thrown away. The epilogue guards `m < M`
  (qmatmul_gpu.mojo:939, 1022) so the extra rows are computed and discarded.

Conclusion: **throughput/datacenter design**, correct and efficient at large M,
structurally wrong for decode (M=1). This is the primary cause of the measured
6-15% of memory roofline: a memory-bound GEMV is being run as a compute-tiled
GEMM that reads the weights once but wastes the tensor-core array and the A
pipeline.

## 2. Is group_size=32 (GGUF Q4_0) second-class on the GPU path?

Yes. The GPU quant path is built around **group_size=128** (GPTQ/AWQ); g32
(Q4_0) is served only by the generic fallback and crashes on every tuned config.

- The tuned dispatch cascade keys **only on static K and N** and ignores
  `group_size` (qmatmul_gpu.mojo:1896, 2013, 2130, 2224). Several branches pick
  `block_tile_shape[2]` (BK) = **128**: 4096x4096 for m<=32
  (qmatmul_gpu.mojo:1901, 1924) and 4096x6144 for m<=32
  (qmatmul_gpu.mojo:2018, 2041). Those are only valid when `group_size >= BK`.
- Inside the kernel the scales layouts use `ceildiv(BK, group_size)` and the
  prefetch cadence uses integer `group_size // BK` (qmatmul_gpu.mojo:374, 529,
  601-603, 806-807). With group_size=32 and BK=128, `group_size // BK == 0` and
  `K // group_size` scale bookkeeping collapses to a zero-sized layout dim →
  comptime "address is out-of-bounds" in `IntTuple.__getitem__`. So any g32 call
  reaching a BK=128 branch **fails to compile** (this is failure A).
- The only g32-safe config in the file is `default_config` (BK=32,
  qmatmul_gpu.mojo:2341); every tuned config with BK=128 assumes g128.
- The repack ops confirm the bias: GPU GPTQ repack is registered **g128 only**
  (`GPTQ_gpu_repack_b4_g128`, quantization.mojo:675; `_desc_act` variant :705),
  and `repack_Q4_0_for_sm8x` hard-codes `group_size = 32`
  (qmatmul_gpu.mojo:1193) as a separate, GGUF-specific entry.
- Both `qmatmul_b4_g32` and `qmatmul_b4_g128` graph ops exist and both call
  `matmul_gpu_qint4` (quantization.mojo:595, 629), so g32 is *wired* on GPU — but
  it inherits the same shape-keyed dispatch, so at any static decode shape it
  either hits an incompatible BK=128 branch (crash) or, when no branch matches,
  falls to `default_config`.

Conclusion: g32 support on GPU is **real but an afterthought** — no g32 tuned
config exists; the tuned table is a g128 table. Failure (A) is therefore a
mixture: an intentional g128 tuning focus **plus** a genuine dispatch bug — the
cascade selects a BK it never checks against `group_size`. A one-line guard
(`BK % group_size == 0`, or skip BK>group_size branches for g32) would turn the
compile crash into a correct (if untuned) run.

## 3. Root cause of the K=14336 crash (B)

Reproduced today on the RTX 3090 with `default_config` (block 128x128x32),
N=4096: **K=4096 (o_proj) passes; K=14336 (down_proj) dies with
CUDA_ERROR_ILLEGAL_ADDRESS** at M=1 and M=8. Same config, same N, only K differs.

Root cause: **the A-tile global load has no M-boundary mask.** `multistage_mma_q`
prefetches a full `[BM, BK] = [128, 32]` A tile every stage via `cp.async`
(`_async_copy_a_tile`, qmatmul_gpu.mojo:305-321; issued at :348, :571; tile taken
at :468/:507 over `a.tiled_iterator[BM, BK, axis=1]`, :823). There is no `m < M`
guard on the load (unlike the epilogue store). With M=1 and BM=128 the copy reads
rows 1..127 of a logically `[1, K]` buffer, i.e. global offsets up to
`(BM-1)*K = 127*K` elements past the start. Whether that overread faults depends
on how far past the allocation it reaches — which scales with **K** (the row
stride):

- K=4096: overread reaches ~127*4096*2 B ≈ 1.0 MB past a 8 KB A buffer → still
  inside mapped/pooled memory → no fault, correct output (only row 0 is stored).
- K=14336: overread reaches ~127*14336*2 B ≈ 3.6 MB past a 28 KB buffer →
  crosses into an unmapped page → illegal access.

Direct proof (not inference): I re-ran down_proj (K=14336, M=1) with the A device
buffer over-allocated to `128*K` (BM rows, zero/garbage-padded) and **the crash
disappeared and the result is correct** (`l2_rel_err = 0.0038`, matching o_proj's
0.0038). Making the phantom rows in-bounds is sufficient to remove the fault,
which pins the cause to the unmasked A-tile overread. (Baseline unpadded run at
the same shape aborts with CUDA_ERROR_ILLEGAL_ADDRESS.)

Classification: **boundary bug, not config-specific and not K-specific in
principle.** It is latent for M < BM at *every* shape; K just determines whether
the overread lands in mapped slack (silently correct, e.g. o_proj) or an
unmapped page (crash, e.g. down_proj). It would also fire for K=4096 given a
tighter allocation. This is almost certainly the same class as the disabled
assertion in `test_multistage_gemm_q` (KERN-2339, open-questions.md item 4:
"vectorized-store OOB").

## 4. Does MAX ship any decode/GEMV-specialized quant path on NVIDIA?

No. On GPU, int4 is **always** the tiled tensor-core GEMM above.

- `linalg/gemv.mojo` (the dedicated GEMV kernel) has **zero** quantized support:
  the only `uint8` in the file is a shared-memory scratch buffer
  (gemv.mojo:2323); no int4/qint/Q4 codepath exists.
- The only GPU int4 entry points are `matmul_gpu_qint4` / `multistage_gemm_q`
  (qmatmul_gpu.mojo:1789, 1659), both of which run `multistage_qgemm_kernel`.
- The CPU int4 kernels (`matmul_qint4`, `matmul_Q4_K` in `qmatmul.mojo`) are a
  separate, CPU-only path used by the float32/no-`DeviceContext` graph ops
  (quantization.mojo:330-336, 427-433); they do not help on GPU.

Contrast with llama.cpp, which has `mul_mat_vec_q` as a dedicated M=1 quantized
GEMV (one thread-block per output row, weights streamed once, no tensor cores,
no wasted M-tile) and hits 62-96% of roofline on the same shapes. **This missing
decode-specialized quant kernel is the structural gap** — the H1 candidate.

## 5. Verdict per issue

| Issue | Design vs. bug | Evidence | Who it affects |
|---|---|---|---|
| **Slowness at M=1** (6-15% roofline, ~6-10x slower than llama.cpp) | **Intentional throughput design** (not a bug). GEMM tiled for large M; M=1 uses 1/128 of the block M-tile and ~127/128 of tensor-core + A bandwidth is wasted. Correct and fast on large-M / datacenter; poor on decode. | mma.sync GEMM: qmatmul_gpu.mojo:485,552; mma shape 16x8x16 tensor_core.mojo:1395; grid `ceildiv(M,BM)` utils_gpu.mojo:242-247; BM=128 default_config qmatmul_gpu.mojo:2341; epilogue `m<M` guard :939. No quant GEMV: gemv.mojo (none). | Anyone doing quantized single-request/low-batch decode on any NVIDIA GPU (consumer or datacenter); just more visible on consumer parts with no cuBLAS-class fallback. |
| **(A) g32 compile crash** on tuned dispatch branches | **Design bias (g128) + genuine dispatch bug.** The tuned table is GPTQ/AWQ g128; picking BK=128 for a g32 call is never guarded, so `group_size // BK == 0` crashes at comptime. Wrong for g32 on any hardware. | BK=128 branches qmatmul_gpu.mojo:1901,1924,2018,2041; `group_size // BK` / `ceildiv(BK,group_size)` :374,529,601,806; g128-only GPU repack quantization.mojo:675,705; only g32-safe config :2341. | Any g32 (GGUF Q4_0) GPU matmul that reaches a tuned (static-K/N) branch — every GPU, not just sm_86. Masked only when the shape misses every branch and falls to default_config. |
| **(B) K=14336 illegal-address crash** | **Genuine bug (wrong on any hardware).** Unmasked A-tile `cp.async` reads `(BM-1)*K` elements past a `[M,K]` buffer when M<BM; faults when the overread (∝K) hits an unmapped page. Padding A to BM rows fixes it (verified). | No `m<M` mask on A load qmatmul_gpu.mojo:305-321,348,571,823; epilogue *does* guard :939,1022; padded-A repro passes (l2 0.0038) vs. unpadded ILLEGAL_ADDRESS. Likely == KERN-2339. | Any quantized GEMM with M<BM (all decode) and a large-enough K/tight-enough allocation — every GPU. On sm_86 it removes the last working Q4_0 down_proj path entirely. |

### Bottom line for Modular

- The **slowness is by design** (throughput GEMM, no decode kernel) — fair to
  frame as "no GPU quant GEMV yet", not a bug. That is the upstreamable gap.
- **(A) and (B) are real bugs**, independent of the design bias, and both are
  cheap to fix: a `group_size`/BK compatibility guard in the dispatch, and an M
  bound (or BM-padded allocation) on the A-tile load. Both are wrong on *any*
  NVIDIA GPU; consumer Ampere just exercises the decode/g32 corner that the
  datacenter-tuned tests do not.
