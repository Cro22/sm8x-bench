# Audit of `modular/max/kernels` for LLM decode on consumer NVIDIA GPUs

- Upstream SHA: `b4497b7ce9ba96331c72c637ad41b44bab374f33` (tag `max/v26.5.0`)
- Mojo version: `Mojo 1.0.0 (ed45d567)`, `max 26.5.0` (installed from PyPI; the
  Modular nightly channels are dead/frozen — see `pyproject.toml`)
- Bench hardware: RTX 3090 (sm_86, driver 610.62, CUDA 13.3). **No RTX 4090
  (sm_89) on this box** — sm_89 claims below are from source gating, not runs;
  sm_89 measurements will be `N/A` until a 4090 box is available.
- Tree paths found:
  - `modular/max/kernels/src/quantization/` — quant matmul (qmatmul_gpu.mojo)
  - `modular/max/kernels/src/linalg/matmul/gpu/` — dense GEMM; `linalg/gemv.mojo`
  - `modular/max/kernels/src/nn/attention/gpu/` — attention (mha.mojo + nvidia/sm90, nvidia/sm100)
  - `modular/max/kernels/src/kv_cache/types.mojo` — paged + continuous KV cache
  - Mojo stdlib gpu host API: `modular/max/mojo/max/gpu/host/device_context.mojo`
    (NOT under `mojo/stdlib/std/gpu/host/`, which only has `info.mojo`)

## Summary

MAX ships **real, tensor-core, consumer-capable** kernels for the two hottest
decode paths: a dedicated M=1 GEMV (SIMT) + a multistage mma.sync GEMM for M>1,
and a genuine flash-decoding split-K attention kernel (`mha_decoding`, FA2,
mma.sync m16n8k16, cp.async). All of these are gated only on
`has_nvidia_gpu_accelerator()` and run unmodified on sm_86/sm_89 — decode is
**not** pushed onto naive/prefill fallbacks. The weakness is **tuning, not
presence**: every config heuristic on the consumer path is A100- or B200-derived
(hard-coded A100 `sm_count`, B200-tuned GEMV thresholds, no sm_8x table row), and
two real format gaps exist — **fp16 M>1 falls out to cuBLAS**, and **weight-quant
GPU matmul is 4-bit-only (Q4_0/GPTQ); Q8_0 and Q4_K have no GPU kernel**. FA3
(wgmma/TMA), fp8 KV, and block-scaled FP8/FP4 matmul are all hard-gated to
sm_90/sm_100. The likely H1 is a config/tuning contribution, not a new kernel —
H0 measurements decide.

## Inventory

### 1. Quantized matmul / GEMV (GPU)

| Kernel / entry point | Path | Formats | Arch gating | Config selection | MMA | Pipelining | Runs sm_86 | Runs sm_89 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `matmul_gpu_qint4` → `multistage_gemm_q` → `multistage_qgemm_kernel` | `quantization/qmatmul_gpu.mojo:1789,1659,654` | int4: GGUF **Q4_0** (g32) + **GPTQ** (g32/128); act = **bf16** (not fp16); weights uint8-packed, bf16 scales | **(a) generic, NVIDIA-only, consumer-capable** — `assert is_nvidia_gpu()` `:696`; kernels named `_for_sm8x` `:1172,1374`; no sm_90/sm_100 gate | **shape-keyed, arch-blind**: hardcoded `comptime if` on (K,N)×M `:1896-2360`; rows only for Llama-3-8B shapes; no A100/H100/consumer row → 3090/4090 get the *same* config as A100. smem-driven stage/partition shrink `:1704-1759` may fire on lower-smem consumer parts | mma.sync **m16n8k16** (bf16→f32) | cp.async, `num_pipeline_stages` 3–5 | yes | yes (from gating) | test `test/gpu/quantization/test_multistage_gemm_q.mojo` runs w/ **assertions off** (KERN-2339 vectorized-store OOB, `BUILD.bazel:102`) — validate correctness carefully |
| Block-scaled FP8/NVFP4/MXFP4/MXFP8 matmul | `linalg/matmul/gpu/sm100/block_scaled_*.mojo`, `sm90/*`, `linalg/mxfp4_matmul_sm90.mojo` | NVFP4, MXFP4, MXFP8, blockwise FP8 | **(c) datacenter-only** — `assert _is_sm10x_gpu` `block_scaled_dispatch.mojo:144,388`; tests → `//:h100_gpu`/`//:b200_gpu` | UMMA/tcgen05/TMA | wgmma/UMMA | **no** | **no** | `N/A (requires sm_90/sm_100)` on bench boxes |
| CPU-only: `matmul_qint4` (Q4_0), `matmul_Q4_K`/`matmul_Q6_K` | `quantization/qmatmul.mojo:1320`, `qmatmul_k.mojo` | Q4_0, Q4_K, Q6_K | CPU only — graph ops assert `is_cpu` `quantization.mojo:432,532` | x86 VNNI / ARM NEON | — | CPU | CPU | **no GPU K-quant matmul exists** |

Dispatch chain for `(M=1, N=4096, K=4096, Q4_0)`: graph op `qmatmul_b4_g32`
(`graph_compiler/builtin_kernels/quantization.mojo:578`, asserts `is_gpu`, a/c
bf16, b uint8) → `matmul_gpu_qint4[32]` (`qmatmul_gpu.mojo:1789`) →
`matmul_gpu_qint4_impl` (`:1842`); K==N==4096 branch `:1896`, m≤16 →
`M16_config` (block 16×64×128, 4 stages, 4 warp-k) → `multistage_gemm_q`
(`:1659`, smem check `:1703`) → `multistage_qgemm_kernel` (`:654`) →
`multistage_mma_q` (`:105`) cp.async + `TensorCore.mma()` m16n8k16 → warp
split-K reduce (`:886`) → bf16 store. One-time weight repack:
`GGUF_gpu_repack_q4_0` → `repack_Q4_0_for_sm8x` (`qmatmul_gpu.mojo:1172`).

### 2. Dense matmul at small M (fp16/bf16)

| Kernel / entry point | Path | Dtypes | Arch gating | Config selection | MMA | Pipelining | Runs sm_86 | Runs sm_89 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **GEMV** `gemv_split_k` (M=1 path) via `gemv_gpu` | `linalg/gemv.mojo:505,1551`; dispatch `matmul/gpu/__init__.mojo:1353` | fp16, bf16, fp32, fp8_e4m3 | **(a) generic**, no sm_8x gate | `_nvidia_gemv_config` `gemv.mojo:999` — **docstring "for NVIDIA B200 GPUs"**, applied verbatim to all non-AMD. (M=1,4096²,fp16)→ threads=128, tile_n=4, unroll=1, grid (1,1024) | **none** (SIMT FMA + warp shuffle) | vectorized non-temporal loads; no cp.async | yes | yes | needs `transpose_b=True` (`:1626`); else cliff to `GEVM`/naive |
| **Tiled GEMM** `multistage_gemm_kernel` (M>1) | `linalg/matmul/gpu/_multistage_gemm_gpu.mojo:702`; launcher `matmul/gpu/__init__.mojo:1576` | **{fp32(TF32), bf16} only** — `matmul_supported_format_nvidia` **excludes fp16** `:517-521` | **(b) generic but A100-tuned** | `select_config` `utils_gpu.mojo:417` hard-codes `A100.sm_count=108` `:488,521`; largest tile `ampere_256x128_3` **disabled off-A100** `:476`; sm_8x picks from 2 tiles | mma.sync **m16n8k16** (bf16), m16n8k8 (TF32) | cp.async, num_stages=4 | yes | yes | per-shape tuned table `create_matmul_configs_ampere` reached **only if device==A100** `:1301` — **(c)** for sm_8x |
| fp16 M>1 → **cuBLAS** vendor fallback | `matmul/gpu/__init__.mojo:1364-1378` | fp16 | vendor | cuBLAS | — | — | via cuBLAS | via cuBLAS | fp16 excluded from Mojo gate ⇒ exits to cuBLAS unless `MODULAR_DISABLE_VENDOR_FALLBACK=1` (→ naive) |

Dispatch chain `(M=1, N=4096, K=4096, fp16, transpose_b)`: `linalg.matmul`
(`matmul/__init__.mojo:46`) → `_matmul_gpu` (`gpu/__init__.mojo:457`); fp16 not
tensor-core-supported so `:722` skipped; `:1353` `m==1` → `_gemv_dispatch` →
`gemv_gpu` (`gemv.mojo:1551`) selects `GEMV_SPLIT_K` (`:1637`) →
`gemv_gpu_dispatch` (`:1170`) → `gemv_split_k` kernel.
`(M=8)` differs by dtype: **bf16** → tensor-core block `:722`, non-A100 `else`
`:1300` → `select_config` → `multistage_gemm[ampere_*]`; **fp16** → cuBLAS
(`:1364`).

### 3. Attention decode (MHA/GQA over KV cache)

KV dtypes on sm_8x: **fp16 / bf16 (fp32 possible)** — fp8 KV requires
`_is_sm10x_gpu` (`mha.mojo:684-689`), so `N/A` on bench boxes. No int4/int8
KV-quant attention kernel exists. split-K over sequence: **yes**. GQA 32/8: q
heads sharing a KV head packed into the M tile; grid.y = kv_heads = 8.

| Kernel / entry point | Path | KV dtypes | Decode-specialized | Arch gating | Config selection | MMA | Pipelining | Runs sm_86 | Runs sm_89 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `mha_decoding` + `mha_splitk_reduce` (**the sm_8x decode path**) | `nn/attention/gpu/mha.mojo:4305,5970` | fp32/fp16/bf16 | **yes** — flash-decoding split-K, online softmax `:4349-4355`; selected when `is_token_generation` `:2257,1391` | **(a) generic** — `has_nvidia_gpu_accelerator()` `:1363`; FA3 disabled unless sm90/sm100 `:1541` | fixed FA2: BM16/BN128/BK32, 4 warps, 4 stages `:1407-1435`; only split-K *partition count* adapts (`cuda_mha_decoding_num_partitions` `mha_decode_partition_heuristic.mojo:98`; 3090@seq4096→8, 4090→8) | mma.sync **m16n8k16** | cp.async, 4 stages | yes | yes | genuine tensor-core decode kernel — not a fallback |
| `mha` / `mha_single_batch` (prefill) | `mha.mojo:2605` | same | no (prefill) | (a) generic | fixed | mma m16n8k16 | cp.async | yes | yes | fork when `is_token_generation==False` |
| SM90 FA3 (`mha_sm90_dispatch`) | `nn/attention/gpu/nvidia/sm90/` | + | — | **(c)** `is_sm90 == H100` `:996` | wgmma/TMA | FA3 | **no** | **no** | sm_8x never reaches |
| SM100 (`mha_sm100_*_dispatch`, MLA, fp8 KV, depth512) | `nn/attention/gpu/nvidia/sm100/` | +fp8 | — | **(c)** `_is_sm10x_gpu` `:997` | tcgen05/TMA | — | **no** | **no** | carries fp8-KV + depth-512 decode |

Dispatch chain `(batch1, q32, kv8, depth128, seq4096, KV fp16)`: op
`generic_flash_attention_kv_cache_ragged` (`nn/kv_cache_ragged.mojo:3756`) →
`_flash_attention_dispatch` (`:3854`) → `gpu_flash_attention[ragged]` (`:3910`)
→ `flash_attention[cache_t: KVCacheT]` (`mha.mojo:576`, dtype asserts
`:691-705`) → `flash_attention_dispatch` (`:867`); `is_token_generation=True`,
not sm90/sm100 → decode `elif` `:1391`; FA2 config `:1407`; partitions=8
(`:1523`); `launch_mha_decoding` (`:1585`) → `mha_decoding` (`:4305`) +
`mha_splitk_reduce`. Paged vs contiguous KV does **not** change the kernel — both
wrap into an `MHAOperand`/`KVCacheT` and hit the same dispatch.

### 4. Paged KV cache

- **Types:** `PagedKVCache` (`kv_cache/types.mojo:1985`), `PagedKVCacheCollection`
  (`:3148`); non-paged `ContinuousBatchingKVCache` (`:1436`).
- **Page/block size:** compile-time struct param `page_size` (`:1988`), exposed
  as a runtime knob frozen at graph build. **Default 128** and **enforced
  `page_size % 128 == 0 and >= 128`** (`max/python/max/pipelines/kv_cache/registry.py:50`).
  → `bench/shapes.py PAGE_SIZE = 128`.
- **Layout:** 6D row-major parent `[total_num_blocks, 2(kv), num_layers,
  page_size, num_heads, head_size]` (`:3186`); per-layer 4D view
  `[total_num_blocks, page_size, num_heads, head_size]` (`:2025`). **`head_size`
  is the contiguous (stride-1) dim.** block stride is runtime (`UNKNOWN_VALUE`).
- **Dtypes:** generic fp16/bf16/fp32/fp8; optional per-block fp8 quant with a
  parallel scales tensor + dequant-on-`load` (`:2089,2724`). Continuous variant
  does **not** support quant (`:1462`).
- **Consumed directly by attention** — no copy-to-contiguous step.
  `KVCacheMHAOperand` (`nn/attention/mha_operand.mojo:377`) forwards
  `row_idx`/`load`/`store`/`block_paged_ptr` straight to the cache; the generic
  pointer/`load` path (`types.mojo:2707-2779`) is what sm_8x uses.
- **Append path:** written during fused QKV matmul
  (`nn/kv_cache_ragged.mojo:535`) via `PagedKVCache.store` (`:2734`):
  `divmod(tok, page_size)` → LUT block lookup → store; sentinel guard skips
  unassigned slots (`:2744`).
- **Arch gating:** generic **(a)** for the whole data structure + store/load
  (runs on sm_86/89). The TMA fast page→smem loader
  (`PagedRowIndices.tma_copy_k/v`, `create_tma_tile`, `:426-924`) is **(c)**
  SM90/SM100 only (`cp.async.bulk.tensor`). No Ampere/Ada-specific paged loader.
- **Tests:** `test/gpu/kv_cache/test_batch_kv_cache_flash_attention*`,
  `test_mha_decoding_vs_naive`, `test_paged_*` — generic-GPU, runnable on one
  consumer GPU. SM100/B200-only spike tests are not.

### 5. Supporting kernels (note only)

RoPE, RMSNorm, SiLU/gate fusion, sampling live under `nn/` and are not
benchmarked in H0. Fused QKV+cache write is `_fused_qkv_matmul_kv_cache_ragged`
(`nn/kv_cache_ragged.mojo:535`).

## Config tables and consumer fallback

Every tuning decision on the sm_8x path is derived from a datacenter part:

- **Dense GEMM** `select_config` (`linalg/matmul/gpu/utils_gpu.mojo:417`): wave
  /occupancy model hard-codes `A100.sm_count = 108` (`:488,521`) regardless of
  the real 82 (sm_86) / 128 (sm_89). Largest tile `ampere_256x128_3` disabled
  unless device == A100 (`:476-478`). The per-shape tuned table
  `create_matmul_configs_ampere` (`sm80/dispatch.mojo:21`) is reached **only** if
  `device == A100` (`matmul/gpu/__init__.mojo:1301-1306`) — sm_8x never uses it.
- **GEMV** `_nvidia_gemv_config` (`linalg/gemv.mojo:999`): docstring "for NVIDIA
  B200 GPUs. B200 has 160 SMs"; thresholds swept on B200, applied verbatim to
  sm_8x. No consumer validation.
- **int4 quant GEMM**: configs keyed purely by (K,N)×M
  (`qmatmul_gpu.mojo:1896-2360`), identical for sm_8x and A100/H100 — arch-blind.
- **Attention decode**: fixed FA2 tile (BM16/BN128/BK32, 4 warps, 4 stages,
  `mha.mojo:1407`); only split-K partition count adapts to SM count.

No GPU-name lookup table has an RTX 3090 or RTX 4090 row anywhere in the decode
path. This is the central finding: **the kernels run on consumer Ampere/Ada, but
nothing in-tree is tuned for it.**

## Compile / run log on this box

- Toolchain compiles and launches GPU code on sm_86: verified by
  `bench/mojo/bw_probe.mojo` (a streaming-read reduction) compiling and running
  on the 3090 — see `reports/h0-results.md` for the measured-roofline number.
  This substantiates "sm_86 can compile + launch mma-free Mojo GPU kernels".
- Per-kernel `mojo test` / minimal-launch runs of the specific MAX kernels above
  (gemv, multistage_gemm, mha_decoding, qint4) are the first step of the
  measurement phase; results and any compile/runtime errors will be logged here
  and in `reports/api-drift.md` as they are attempted. **Not yet attempted at
  audit time** (blocked on the harness entry points, not on any gating).

## Gaps and candidates for H1 (ranked)

Headroom columns are filled after H0 measurements; do not presuppose.

1. **Consumer config tuning for dense GEMM + GEMV (config fix, no new kernel).**
   Evidence: A100-hardcoded occupancy `utils_gpu.mojo:488,521`; A100-only tuned
   table `:1301`; B200-tuned GEMV `gemv.mojo:999`. Fix type: **config-table /
   heuristic PR** adding sm_86/sm_89 rows. No tensor cores needed beyond what
   exists. Headroom: TBD from H0 (% of measured roofline for GEMV M=1 and GEMM
   M=8 vs cuBLAS/llama.cpp). This is the most likely H1 and the cheapest.
2. **int4 quant GEMM consumer tuning + correctness.** Arch-blind config
   (`qmatmul_gpu.mojo:1896`) + smem-shrink risk on lower-smem parts (`:1704`) +
   the KERN-2339 store bug (assertions off in test). Fix type: config + a
   correctness fix. Needs tensor cores (already mma m16n8k16). Headroom: TBD vs
   llama.cpp Q4_0 `mul_mat_vec_q`.
3. **fp16 M>1 exits to cuBLAS — no Mojo kernel.** `matmul_supported_format_nvidia`
   excludes fp16 (`matmul/gpu/__init__.mojo:517`); the multistage kernel already
   supports fp16 mma (`tensor_core.mojo:1401`). Fix type: **enable fp16 in the
   dispatch gate** (small) — but only worth it if the Mojo kernel beats cuBLAS on
   sm_8x, which H0 measures. Record fp16 M>1 numbers as cuBLAS.
4. **Attention decode config for sm_8x.** Fixed FA2 tile, never swept for
   consumer (`mha.mojo:1407`); no FA3 on sm_8x by design. Fix type: config
   sweep, possibly a cp.async-staged paged loader (the TMA one is sm_90+ only).
   Needs measurement vs FlashInfer to know if BM16/BN128/4-stage leaves headroom.
5. **Format gaps that are correctly `N/A` (not H1 targets, just report them):**
   Q8_0 / Q4_K have no GPU matmul (CPU-only); block-scaled FP8/FP4 and fp8 KV are
   sm_90/sm_100-only. These bound what MAX can be benchmarked on: **MAX GPU
   weight-quant = Q4_0 only; MAX GPU KV = fp16/bf16 only.**

## Impact on the benchmark plan (`bench/shapes.py`, H0 DoD #2)

- `PAGE_SIZE = 128` (set).
- **MAX GPU coverage is narrower than the llama.cpp baseline set.** Per impl:
  - MAX GPU GEMV/matmul: **fp16** (M=1 native GEMV; M=8 → cuBLAS), **bf16**
    (M=1 GEMV; M=8 native GEMM), **Q4_0** (int4 kernel, bf16 activations). MAX has
    **no** GPU Q8_0 and **no** GPU Q4_K → those cells are `N/A (CPU-only)` for MAX
    while llama.cpp fills them. `WEIGHT_FORMATS` in shapes.py is the union across
    impls; the per-impl support matrix above governs which cells are `N/A`.
  - MAX attention decode: KV **fp16/bf16** only.
- Note the **activation-dtype asymmetry**: MAX int4 quant GEMM takes **bf16**
  activations, not fp16 — seed the same bytes but cast appropriately and record
  it in the results JSON `dtype` block.

## Open questions for the forum

Listed in `reports/open-questions.md`. Titles: (1) is the A100-hardcoded
`sm_count` in `select_config` intentional for consumer parts or an oversight;
(2) is fp16 excluded from `matmul_supported_format_nvidia` deliberately (always
prefer cuBLAS for fp16) or is the Mojo kernel expected to handle it; (3)
KERN-2339 status and whether it affects Q4_0 decode shapes.
