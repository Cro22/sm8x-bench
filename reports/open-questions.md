# Open questions for forum.modular.com

Drafts only. Never posted by Claude Code.

## For the forum (from the H0 audit, max/v26.5.0)

1. **A100-hardcoded `sm_count` in dense-GEMM config selection.**
   `select_config` uses `A100.sm_count` (108) for its wave/occupancy model
   (`linalg/matmul/gpu/utils_gpu.mojo:488,521`) even when the device is an RTX
   3090 (82 SMs) or 4090 (128 SMs). And the per-shape tuned table
   `create_matmul_configs_ampere` is reached only when `device == A100`
   (`matmul/gpu/__init__.mojo:1301`), despite its "Ampere/SM80" name. Is consumer
   Ampere/Ada meant to fall back to the A100 model deliberately, or is this an
   oversight? (Would a sm_86/sm_89 row be welcome upstream?)

2. **fp16 excluded from `matmul_supported_format_nvidia`.** fp16 M>1 exits the
   Mojo path to cuBLAS (`matmul/gpu/__init__.mojo:517-521,1364`), though the
   multistage kernel supports fp16 mma (`layout/tensor_core.mojo:1401`). Is fp16
   intentionally always routed to cuBLAS, or is enabling the Mojo kernel for fp16
   in scope?

3. **`_nvidia_gemv_config` tuned "for NVIDIA B200".** The GEMV config
   (`linalg/gemv.mojo:999`) is documented as B200-tuned and applied to all
   non-AMD GPUs. Was it validated on consumer Ampere/Ada, or is a consumer sweep
   welcome?

4. **KERN-2339.** `test_multistage_gemm_q` runs with assertions disabled due to a
   vectorized-store OOB (`test/gpu/quantization/BUILD.bazel:102`). Status? Does it
   affect the Q4_0 decode shapes (K=N=4096, etc.)?

5. **`matmul_gpu_qint4` tuned configs are incompatible with group_size=32 (GGUF
   Q4_0).** For a static shape like 4096x4096 with m<=32, the dispatch
   (`qmatmul_gpu.mojo:1896+`) selects a config with `block_tile_shape[2]` (BK) =
   128. Inside the kernel `group_size // BK` = `32 // 128` = 0, producing a
   zero-sized scales-layout dimension and a comptime failure ("address is
   out-of-bounds" in `int_tuple.__getitem__`). These configs appear tuned for
   group_size=128; the group_size=32 (GGUF Q4_0) path only works via the generic
   `default_config` (BK=32, 128x128 tile). Is g32 meant to be supported through
   this entry on GPU, or only g128? On sm_86 this leaves Q4_0 with no working
   M=1-specialized config — decode-shape Q4_0 runs a 128x128 GEMM tile at ~6% of
   the memory roofline (~4x slower than the fp16 GEMV). Repro shape + measurement
   in bench/results; details in reports/api-drift.md.

## Internal (methodology, not for forum)

- **Decode-attention byte formula wording.** `bench-methodology` SKILL writes the
  minimum KV traffic as `seq_len * kv_heads * head_dim * 2 (K) * 2 (K and V) *
  bytes`, i.e. two factors of 2 (4x). The algorithmic minimum is one read of K
  plus one read of V = 2x. `bench/roofline.py:attention_decode_bytes` uses 2x.
  Confirm with Jesús that the skill wording is a typo, then fix the skill text.
  (Raised 2026-09-01.)

## matmul_gpu_qint4 broken for Q4_0 (group_size=32) with static N/K (sm_86, max 26.5.0)

The public wrapper `quantization.qmatmul_gpu.matmul_gpu_qint4[group_size=32,
target="gpu"]` fails to compile for static shapes that hit a tuned dispatch
branch (e.g. 4096x4096, m<=32). Those branches use `block_tile_shape[2]` (BK) =
128, but Q4_0 has group_size=32, so `group_size // BK == 0` produces a zero-sized
scales-layout dimension and a comptime crash ("address is out-of-bounds" in
`layout::int_tuple::IntTuple::__getitem__`, via `multistage_qgemm_kernel`
smem setup). The tuned configs appear to assume group_size=128 (GPTQ/AWQ). The
BK=32 `default_config` works and is correct (L2 3.8e-3), but there is no working
M=1 GEMV-tuned config for Q4_0. Q: is `matmul_gpu_qint4[32]` expected to work on
consumer Ampere for Q4_0, or is GGUF Q4_0 only wired through the graph compiler
with dynamic shapes (which then also fail the "Layout must be fully static"
constraint on B here)? Repro: bench/mojo/qgemv_max.mojo history + api-drift.md.

Follow-up (measured): even the working BK=32 `default_config` **crashes with
CUDA_ERROR_ILLEGAL_ADDRESS for K=14336** (Llama-3 down_proj, N=4096) at
group_size=32 on sm_86 — so that shape has no working Q4_0 GPU path at all. The
shapes that DO run (K=4096) measure 6-15% of the memory roofline (166 us at
N=4096 up to 2094 us at N=128256), ~4x slower than the fp16 GEMV. See
bench/results/max_gemv_Q4*.json.

ROOT CAUSE (see reports/max-q4_0-analysis.md). IMPORTANT CORRECTION after
measuring MAX's REAL dispatch (not a forced config):
- **Compile failure (A) — real, cheap fix.** For static shapes whose tuned
  config uses BK=128 (o_proj K4096/N4096 and qkv K4096/N6144 at m<=32), dispatch
  has no `group_size % BK` guard, so group_size=32 hits `32 // 128 == 0` → a
  zero-sized scales-layout dim → comptime crash. The tuned table is a GPTQ/AWQ
  g128 table; g32 (GGUF Q4_0) is second-class there. MAX genuinely cannot compile
  Q4_0-g32 for o_proj/qkv. (up_proj/down_proj have BK=32 configs and compile fine.)
- **The K=14336 "crash" (B) is a LATENT bug, NOT reached by MAX's real dispatch.**
  The A-tile `cp.async` load in `multistage_mma_q` has no `m < M` mask, so with
  M < BM it over-reads `(BM-1)*K` past the `[M,K]` A buffer and faults when that
  crosses an unmapped page. We hit it only because our FIRST harness *forced*
  BM=128; MAX's real dispatch for down_proj uses **BM=16**, which runs cleanly
  (43.6 us, 81% roofline — no crash). So the OOB is a genuine latent defect
  (wrong on any GPU when a BM>M config is selected at large K) but does NOT
  manifest for these Llama-3 shapes under the shipping dispatcher. Our earlier
  "MAX crashes on down_proj" claim was an artifact of the forced config and is
  RETRACTED.
- **Slowness is design, and only on some shapes.** The kernel is a tensor-core
  GEMM tiled for large M. Shapes with a decode-tuned config (up_proj/down_proj,
  BM=16) reach 69-81% of roofline — competitive. Shapes that fall to the default
  128x128 tile (gate_up/lm_head) run at ~15% at M=1 (1/128 M-tile utilization).
  MAX has no single decode-quant kernel that is uniformly good; that is the
  design gap, not a bug.
