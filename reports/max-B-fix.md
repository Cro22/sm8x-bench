# Fix B — mask the A-tile load in `multistage_mma_q` (unmasked over-read when M < BM)

PR/issue-ready writeup for upstream (`modular/max/kernels`). Nothing here edits the
read-only `modular/` submodule; this is the change to make in a separate clone of
`modular/main`. **Not yet built/tested** — it must be compiled and run under ASan
in a MAX build env; see "Build & test" below. Coordinate with the in-flight
[#6708](https://github.com/modular/modular/pull/6708) rework of this same file.

## Summary

The quantized multistage matmul copies a `BM × BK` A-tile (activations) from global
to shared memory with **no runtime `M` bound**. When a selected config has `BM > M`
(true for every decode shape: `M = 1`, `BM ∈ {16, 32, 128}`), the async copy reads
`(BM − M_rem) × K` elements past the end of the `[M, K]` A buffer. It faults
(`CUDA_ERROR_ILLEGAL_ADDRESS`) when that over-read crosses an unmapped page — which
we hit at `K = 14336` with a forced `BM = 128`. The shipping Llama-3 dispatch uses
`BM = 16` for that shape, so the over-read lands in mapped padding and does not
fault today — but the read is still out of bounds and is a latent correctness bug on
any input where the tail tile runs off a mapped page.

The **dense** multistage GEMM does not have this bug: it bounds the tail copy with
`_mask_tensor_row`. The quantized path was adapted from it and dropped that masking.
The fix is to port the same masking to the quant A-copy.

## Reproduction (measured, this repo)

- Harness `bench/mojo/qgemv_max.mojo` forcing `multistage_gemm_q` at `BM = 128`,
  shape `N = 4096, K = 14336, M = 1`, `group_size = 32` → `CUDA_ERROR_ILLEGAL_ADDRESS`.
- The real dispatcher (`bench/run_gemv_max_dispatch.py`) picks `BM = 16` for that
  shape and runs clean (43.5 µs) — the over-read exists but doesn't fault. Both are
  in `bench/results/` (down_proj rows).

## Root cause (verified file:line)

`max/kernels/src/quantization/qmatmul_gpu.mojo`:

- A-tile iterator is created with no `M` bound (v26.5.0 line 823; **main line 737**):
  ```mojo
  var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx[1], bk_start)
  ```
- In `multistage_mma_q`, the A copy `_async_copy_a_tile(...)` (v26.5.0 ~348-353;
  prefetch, plus the steady-state and drain copies) passes `a_iter[]` straight to
  `GenericToSharedAsyncTileCopier(...).copy(dst, src)` with no row predicate. The
  copier copies all `BM` rows regardless of `M`.
- The epilogue *does* guard stores with `m < M`, but the **prefetch/mainloop A load
  does not** — so the store is correct, the load is not.

For contrast, the dense path already masks the tail. In
`max/kernels/src/linalg/matmul/gpu/_multistage_gemm_gpu.mojo`:
- helper `_mask_tensor_row(tensor, num_rows)` (line 300) rebuilds the tensor with a
  `RuntimeLayout` whose row extent is `num_rows`;
- every B copy wraps the source: `_mask_tensor_row(b_iter[], num_rows_bound)` with
  `num_rows_bound = max(0, num_b_rows.value() − stage * BK)` (lines 354-367, 493-,
  600-, 640-), threaded from an `Optional[Int]` `num_b_rows` kernel argument.

## The fix

Mirror the dense path's masking for the quant A-copy:

1. Thread the runtime row count into the mainloop. Add
   `num_a_rows: Optional[Int] = None` to `multistage_mma_q` (alongside the existing
   config params), and pass `Int(m)` from `multistage_gemm_q` at the call site
   (v26.5.0 `qmatmul_gpu.mojo:862`, where `m` is already in scope).
2. Add a local `_mask_tensor_row` (copy the dense helper, or hoist it to a shared
   util so both kernels use one copy).
3. At **each** A copy site (prefetch, steady-state, drain) wrap the source before
   `_async_copy_a_tile`:
   ```mojo
   # A is [M, K] laid out non-transposed → the M-tile bound decreases by BM per
   # M-block; within a fixed M-block (block_idx[1]) the K-stage does not change the
   # valid-row count, so the bound is simply the rows of this M-tile that are < M.
   var a_src = a_iter[].bitcast[a_type, target_address_space=AddressSpace.GENERIC]()
   comptime if a_num_rows_known_lt_BM:      # or always, guarded by `if num_a_rows:`
       var rows = max(0, num_a_rows.value() - block_idx[1] * BM)   # rows of this tile ≤ BM
       a_src = _mask_tensor_row(a_src, min(rows, BM))
   _async_copy_a_tile(a_smem_tile, a_src)
   ```
   (Exact bound: the A-tile for `block_idx[1]` covers rows `[block_idx[1]*BM,
   block_idx[1]*BM + BM)`; the valid count is `clamp(M - block_idx[1]*BM, 0, BM)`.
   Unlike B, the A bound does **not** depend on the K-stage, so no `stage*BK` term.)
4. Verify `GenericToSharedAsyncTileCopier.copy` honors the source's reduced runtime
   row extent (zero/skip rows ≥ bound). If it iterates by the source runtime shape
   it already does; if it uses the comptime `BM`, the copier needs the same
   predication the dense `copy_dram_to_sram` path uses. **This is the one point that
   must be confirmed by building** — it decides whether the fix is "wrap the source"
   (above) or "add row predication in the copier".

Guarding the whole thing on `if num_a_rows:` keeps the fast path (M a multiple of
BM, or `num_a_rows=None`) unchanged.

## Test

Add to `max/kernels/test/gpu/quantization/`:
- A `matmul_gpu_qint4` (or `multistage_gemm_q`) case with `M = 1`, a config whose
  `BM = 128`, and `K = 14336`, run under `--config=asan` (or with a guard page after
  the A allocation) so the over-read is caught; assert (a) no OOB and (b) output
  matches an fp32 dequant reference (L2 < 3e-2).
- A correctness case at `M` not a multiple of `BM` (e.g. `M = 3`, `BM = 16`) to lock
  the tail-row masking.

## Build & test (not done here)

This box has no MAX bazel build env, and building the monorepo under WSL is out of
scope for the bench workbench. The change must be built and tested where MAX is
built:
```
./bazelw test --config=asan //max/kernels/test/gpu/quantization/...
```
Do **not** land or report this as fixed until that passes.

## Coordination

[#6708](https://github.com/modular/modular/pull/6708) (open, draft) is actively
reworking `multistage_mma_q` for NVFP4 and touches the same copy sites (it does
*not* add A-row masking — verified). File this as an issue referencing the repro,
or a PR based on #6708's branch, so the two don't conflict. The dense-path masking
pattern above is the reference implementation to copy.
