# Ready-to-post GitHub issue for modular/modular

Copy-paste the block below as a new issue at github.com/modular/modular.
**Claude does not post this** — no auth on this box, and per repo policy outward-
facing posts are yours to make. Detailed fix spec: `reports/max-B-fix.md`.

---

**Title:** `[Kernels][GPU] multistage_mma_q: A-tile load is unmasked for M < BM (out-of-bounds global read; illegal-address at large K)`

**Labels (suggest):** bug, kernels, gpu

---

### Summary

In the quantized multistage matmul (`max/kernels/src/quantization/qmatmul_gpu.mojo`),
the A-tile (activations) is copied global→shared as a full `BM × BK` tile with **no
runtime `M` bound**. When the selected config has `BM > M` (every decode case:
`M = 1`, `BM ∈ {16, 32, 128}`), the async copy reads `(BM − M_rem) × K` elements past
the end of the `[M, K]` A buffer. It is out of bounds always; it *faults*
(`CUDA_ERROR_ILLEGAL_ADDRESS`) when the over-read crosses an unmapped page.

The dense multistage GEMM does not have this — it bounds the tail copy with
`_mask_tensor_row`. The quantized path appears to have been adapted from it without
that masking.

### Environment

- `max` 26.5.0 (also present on `main` as of 2026-09-03).
- NVIDIA RTX 3090 (sm_86), driver 610.62, CUDA 13.3. GGUF Q4_0, `group_size = 32`.

### Repro

- Forcing `multistage_gemm_q` with `BM = 128` at `N = 4096, K = 14336, M = 1`
  (Llama-3 down_proj) → `CUDA_ERROR_ILLEGAL_ADDRESS`.
- The shipping dispatcher selects `BM = 16` for this shape, so the over-read lands
  in mapped padding and does **not** fault today (runs ~43 µs). The read is still
  out of bounds — it will fault on any input whose tail tile runs off a mapped page,
  and trips ASan.

### Root cause (line numbers on `main`)

- `qmatmul_gpu.mojo:737` — `var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_m, k_tile_start)`
  has no `M` bound.
- In `multistage_mma_q`, the A copy passes `a_iter[]` straight into
  `GenericToSharedAsyncTileCopier(...).copy(dst, src)` (prefetch + steady-state +
  drain) with no row predicate; all `BM` rows are read.
- The epilogue guards stores with `m < M`, but the load does not.

### Suggested fix

Mirror the dense path, which already masks the tail copy:
`_multistage_gemm_gpu.mojo:300` (`_mask_tensor_row(tensor, num_rows)`) and the B
copies at `:354-367` wrapping the source with
`num_rows_bound = max(0, num_b_rows − stage*BK)`.

For the quant A-copy: thread an `Optional[Int]` row count (`= M`) into
`multistage_mma_q`, and wrap the A source at each copy site with
`_mask_tensor_row(a_src, clamp(M − block_m*BM, 0, BM))` (A's bound does not depend on
the K-stage, unlike B). Guard on `if num_a_rows:` to leave the M-multiple-of-BM fast
path unchanged. One point to confirm while building: whether
`GenericToSharedAsyncTileCopier.copy` honors a reduced source row extent, or whether
it needs explicit row predication like the dense `copy_dram_to_sram` path.

Happy to open a PR. Note: [#6708](https://github.com/modular/modular/pull/6708) is
reworking this same file for NVFP4 and does **not** add A-row masking — worth
coordinating so the fix isn't lost in the rebase.

### Test

A `matmul_gpu_qint4` case with `M = 1`, a `BM = 128` config, `K = 14336`, run under
`--config=asan`, asserting no OOB and output within L2 < 3e-2 of an fp32 dequant
reference; plus an `M = 3, BM = 16` case to lock the tail-row masking.
