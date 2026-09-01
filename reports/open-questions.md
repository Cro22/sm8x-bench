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

## Internal (methodology, not for forum)

- **Decode-attention byte formula wording.** `bench-methodology` SKILL writes the
  minimum KV traffic as `seq_len * kv_heads * head_dim * 2 (K) * 2 (K and V) *
  bytes`, i.e. two factors of 2 (4x). The algorithmic minimum is one read of K
  plus one read of V = 2x. `bench/roofline.py:attention_decode_bytes` uses 2x.
  Confirm with Jesús that the skill wording is a typo, then fix the skill text.
  (Raised 2026-09-01.)
