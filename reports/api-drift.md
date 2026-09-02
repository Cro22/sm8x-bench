# API drift log

`old → new (mojo version, file where it bit us)`

## Mojo 1.0.0 / max 26.5.0

- `from gpu.host import DeviceContext` → `from max.gpu.host import DeviceContext, DeviceBuffer`
  (host API moved under `max.gpu.host`; source is
  `modular/max/mojo/max/gpu/host/device_context.mojo`, NOT `mojo/stdlib/std/gpu/host/`
  which now only holds `info.mojo`). Confirmed via OSS kernel tests
  (`modular/max/kernels/test/gpu/linalg/*` import lines).
- `from gpu import block_idx, thread_idx, ...` → `from std.gpu import block_idx,
  thread_idx, block_dim, grid_dim, global_idx, WARP_SIZE` (GPU intrinsics live
  under `std.gpu`).
- Toolchain is `uv run mojo` in this repo (NOT `pixi run mojo` as the skills
  say) — the `modular` package is installed as a uv/PyPI wheel; `mojo` resolves
  to `.venv/bin/mojo`.
- **`fn` has been REMOVED entirely** → use `def` for everything (functions and
  GPU kernels). The compiler errors: `'fn' has been removed; use 'def' instead`.
  Confirmed: OSS source at this SHA has 15003 `def` decls vs 1 `fn`. This is the
  single biggest change to adapt to; every skill/example that shows `fn` is stale.
- `def` is **non-raising by default** → a `def` that calls any raising API
  (`DeviceContext()`, `enqueue_*`, `map_to_host`, `execution_time`) must be
  `def foo(...) raises:`. Pure GPU kernels typically don't need `raises`.
- `List` is **not implicitly copyable** → `var v = other_list` fails; use
  `var v = other_list.copy()` or transfer with `^`.
- Device timing: `ctx.execution_time[closure](num_iters) -> Int` (nanoseconds,
  CUDA-event based) is the public timer — preferred over host `perf_counter_ns`.
  Closure form that compiles: `@parameter @always_inline @__copy_capture(...)
  def launch(ctx: DeviceContext) raises: ctx.enqueue_function[k](...)`.
  (NOTE: `@__copy_capture`/`@parameter` is the legacy closure idiom; the
  closure_migration skill wants value-taking unified closures — revisit if a
  newer `execution_time` overload accepts one. Works as-is for now.)
- Confirmed unchanged: `ctx.enqueue_create_buffer[dtype](n)`, `.enqueue_fill(v)`,
  `buf.map_to_host() as h:`, `ctx.enqueue_function[kernel](args, grid_dim=,
  block_dim=)`, `ctx.synchronize()`. A `DeviceBuffer[dtype]` arrives in the
  kernel as `Pointer[Int32, MutAnyOrigin]`.
- Import paths (verified): `from std.gpu import global_idx, thread_idx,
  block_dim, grid_dim`; `from std.gpu.globals import WARP_SIZE`;
  `from std.gpu.primitives import warp` (`warp.sum(x)`); `from std.atomic import
  Atomic` (`Atomic.fetch_add(ptr, v)`); `from std.sys import has_accelerator,
  size_of`; `from max.gpu.host import DeviceContext`. `comptime` (not `alias`).

All of the above verified by `bench/mojo/bw_probe.mojo` compiling and running on
the RTX 3090 (sm_86), Mojo 1.0.0 / max 26.5.0.

## From bench/mojo/gemv_max.mojo (calling the MAX matmul/GEMV kernel)

- Kernel: `from linalg.matmul import matmul`; call
  `matmul[transpose_b=True, target="gpu"](c, a, b, ctx)`. `target` is a
  `StaticString` — pass the literal **`"gpu"`**, NOT `get_gpu_target()` (which
  returns an MLIR target type). GPU path uses `target` only for the `is_cpu[]`
  check + tracing; arch/kernel selection comes from `ctx` device info. M=1
  routes to the dedicated GEMV internally. (`matmul/__init__.mojo:107-147`.)
- TileTensor over a device buffer: `from layout import TileTensor, row_major`;
  `var t = TileTensor(dev_buf, row_major(rows, cols))` (`row_major` takes
  runtime `Int`s; default address space is `AddressSpace.GENERIC`, which matmul
  requires). Pattern from `test/gpu/linalg/test_gemv.mojo:142`. Layout:
  a=x[M,K], b=W[N,K] with transpose_b=True, c=y[M,N].
- `open(path, "rb")` → runtime error `invalid mode: "rb"`. Valid modes:
  `{"r","w","rw","a"}`. Use `open(path, "r")`; `read_bytes()` still returns raw
  `List[UInt8]`.
- `memcpy` → deprecated; use `from std.memory import unsafe_memcpy` with
  `dest=`, `src=`, `count=` (count in elements).
- `ptr.bitcast[T]()` → deprecated; use `ptr.unsafe_bitcast[T]()`.
- `ref` is a reserved keyword (origin syntax) — cannot be a variable name.
- `ptr[i]` positional indexing → deprecated (wants `unsafe_offset=`); read into
  a `HostBuffer` and index that instead.

## From kernels/q4_0_gemv.mojo (OUR OWN raw Q4_0 GEMV GPU kernel)

- `Pointer` (the type a `DeviceBuffer` arrives as in a kernel, e.g.
  `Pointer[UInt8, MutAnyOrigin]`) has **no `.offset(i)` method** (compile error
  "value has no attribute 'offset'"). To read a differently-typed scalar at a
  byte offset, bitcast the WHOLE pointer and index in the target dtype's units:
  `w.unsafe_bitcast[Scalar[DType.float16]]().unsafe_load(byte_off // 2)`
  (byte_off must be a multiple of 2; here every Q4_0 block starts on an even
  byte). Load/store on a raw `Pointer`: `p.unsafe_load(i)` (optionally
  `unsafe_load[width=W](i)` for a vector) and `p.unsafe_store(i, val)`.
- Get the kernel-side pointer from a device buffer with `buf.unsafe_ptr()`; pass
  it as a kernel arg and capture it in the timing closure via `@__copy_capture`.
- fp16 bit-reinterpret of raw GGUF scale bytes: the above pointer-bitcast +
  `.cast[DType.float32]()` is the whole recipe — no manual byte assembly needed
  since block offsets are 2-aligned. bf16 activations: `x.unsafe_load(i).cast[
  DType.float32]()`.
- Nibble extraction on `UInt8`: `byte & UInt8(0x0F)` (low), `byte >> 4` (high).
  The `>> 4` on a `UInt8` yields the high nibble directly (no mask needed).
- `warp.sum(acc)` (`from std.gpu.primitives import warp`) reduces a per-lane
  fp32 across the 32 lanes; every lane gets the total, so guard the store with
  `if lane == 0`. `WARP_SIZE` from `std.gpu.globals`.
- Parametric GPU kernel launch: define `def k[N: Int, K: Int, RPB: Int](...)`
  with `from std.gpu import thread_idx, block_idx, block_dim` and launch the
  fully-specialized function object:
  `ctx.enqueue_function[k[N, K, RPB]](args, grid_dim=G, block_dim=B)`. `grid_dim`
  and `block_dim` are plain `Int`s (comptime here). One warp per output row:
  `block_dim = WARP_SIZE * RPB`, `grid_dim = ceildiv(N, RPB)`, row =
  `block_idx.x * RPB + thread_idx.x // WARP_SIZE`.
- Importing our own kernel module: `uv run mojo build <entry>.mojo -I kernels`
  puts `kernels/` on the import path so `from q4_0_gemv import ...` resolves.
  (The "no -I" note only applies to NOT needing `modular/max/kernels/src`.)
- Perf note (sm_86, RTX 3090): 1-warp-per-row, 1-byte-per-lane-per-block Q4_0
  GEMV at 4096x4096 M=1 gives ~32 us (296 GB/s, 32% of 936 spec) at
  ROWS_PER_BLOCK=2; the sweep {1,2,4,8,16} spans 32-38 us (RPB=2 best). vs
  llama.cpp ~17 us and MAX ~166 us. Correctness L2 rel err 3.3e-7 (exact,
  same dequant as the ref).

## From bench/mojo/attn_max.mojo (MAX flash-decoding attention over paged KV)

- Imports (verified): `from kv_cache.types import KVCacheStaticParams,
  PagedKVCacheCollection`; `from nn.attention.gpu.mha import flash_attention,
  mha_gpu_naive`; `from nn.attention.mha_mask import CausalMask`;
  `from layout._utils import ManagedLayoutTensor`; `from layout._fillers import
  random`; `from layout import Layout, LayoutTensor, RuntimeLayout,
  UNKNOWN_VALUE`. All resolve under `-I modular/max/kernels/src`.
- `PagedKVCacheCollection[dtype, kv_params, page_size](blocks, cache_lengths,
  lookup_table, UInt32(max_prompt_length), UInt32(max_full_context_length))`.
  `page_size` is a **comptime** param (3rd), not a ctor arg. Blocks tensor is 6D
  `[num_paged_blocks, 2, num_layers, page_size, num_heads, head_size]`;
  lookup_table is 2D `[batch_size, padded_lut_cols(num_pages)]` uint32.
- `page_size` must be a multiple of 128 (>=128); production default is 256, we
  use 128. `padded_lut_cols(cols) = ((cols+7)//8)*8 + 16` — the LUT row stride
  must be a mult of 8 and >= cols+15 for `PagedKVCache.populate`'s SIMD path
  (mirror of the test util / cache_manager.py; inlined, the util is not on the
  `src` include path).
- Flash-decoding call (decode path is just seq_len=1 per batch item, no separate
  entry point): `flash_attention[ragged=True](output_lt, q_lt,
  kv.get_key_cache(layer_idx), kv.get_value_cache(layer_idx), CausalMask(),
  row_offsets_lt, scale, ctx)`. Q/output are LayoutTensor
  `[total_q_rows, num_q_heads, head_size]`. `flash_attention` sets
  `is_token_generation=True` internally when max_prompt_len==1 and dispatches
  mha_decoding — no `decode=`/`token_generation=` flag to pass.
- Naive reference (KVCacheT overload, mha.mojo:6922) validated against the SAME
  paged cache: `mha_gpu_naive[ragged=True](q_lt, kv.get_key_cache(layer_idx),
  kv.get_value_cache(layer_idx), CausalMask(), ref_output_lt, row_offsets_lt,
  scale, batch_size, max_prompt_length, max_full_context_length, num_q_heads,
  head_size, num_q_heads // kv_params.num_heads /*group*/, ctx)`. Note the
  arg after `output` is `valid_length` but in `ragged=True` mode you pass the
  **row_offsets** tensor there (as the upstream tests do).
- **fp16 works unmodified** for flash_attention + mha_gpu_naive on sm_86
  (RTX 3090); no bf16 fallback needed. L2 rel err flash-vs-naive ~4.5e-4 at
  seq_len=1024, GQA 32/8, head_size 128.
- `ManagedLayoutTensor[dtype, layout](RuntimeLayout[layout].row_major(
  IndexList[N](dims...)), ctx)`. Fill host via `mt.tensor[update=False]()`
  (no device read-back), pass `mt.device_tensor()` to kernels (default
  `update=True` copies host->device + syncs), read results via `mt.tensor()`
  (default `update=True` copies device->host + syncs). Element access on the
  returned LayoutTensor (`t[i,j,k]`) yields a SIMD with a symbolic length —
  index `[0]` to get a scalar before `.cast[...]()` (a bare `Float32(...)`
  conversion fails: "SIMDLength(Layout(...).size()) vs SIMDLength(1)").

## From bench/mojo/qgemv_max.mojo (MAX int4 / Q4_0 quantized matmul)

- Two-step API in `quantization.qmatmul_gpu`: repack raw GGUF Q4_0 bytes ONCE,
  then matmul. `from quantization.qmatmul_gpu import gpu_qint4_repack_Q4_0,
  multistage_gemm_q` and `from linalg.utils_gpu import MatmulConfig`.
- Both operands of `gpu_qint4_repack_Q4_0[target="gpu"](b_raw_tt, b_packed_tt,
  ctx)` are rank-2 **uint8** TileTensors and must have **STATIC** shapes: the
  kernel reads `comptime N = Int(b.layout.shape[0])` / `K` from the layout for
  grid geometry and the packed layout. Build them with `Idx[...]`:
  `TileTensor(buf, row_major(Coord(Idx[N], Idx[(K//32)*18])))` for the raw
  weights and `Coord(Idx[N], Idx[(K*9)//16])` for the packed output.
- Shapes/bytes (group_size=32, 18 B/32-weight block): raw Q4_0 =
  `N*(K//32)*18` bytes, shaped `[N, (K//32)*18]`; packed = `N*K//2` (4-bit
  weights) + `N*(K//32)*2` (bf16 scales) = `N*K*9//16` bytes, shaped
  `[N, (K*9)//16]`. For K%32==0 both are integers. (These two byte totals are
  equal because 18/32 == 9/16.)
- The quantized GEMM kernel (`multistage_qgemm_kernel`) derives **N and K at
  comptime from the packed-B layout** (not from A/C), so N and K must be static
  on B. M is read at runtime (`c.dim[0]()`), so A=[M,K] and C=[M,N] may keep M
  dynamic but need static K/N: `row_major(Coord(M, Idx[K]))` /
  `row_major(Coord(M, Idx[N]))`. A and C are **bfloat16** (asserted); B uint8.
- BUG/LIMITATION (upstream, verified on sm_86, max 26.5.0): the public
  `matmul_gpu_qint4[group_size=32, target="gpu"](c, a, b, ctx)` wrapper is
  UNUSABLE for Q4_0 (group_size=32) with static N/K. Its per-shape tuned
  configs for e.g. static 4096x4096, m<=32 use `block_tile_shape[2]` (BK) = 128;
  inside the kernel `group_size // BK == 32 // 128 == 0` yields a zero-sized
  scales layout dimension and a **comptime failure** ("address is out-of-bounds"
  in `int_tuple.__getitem__`). Those configs were tuned for group_size=128.
  Passing fully-dynamic A/C to dodge the dispatch instead compiles but produces
  GARBAGE (L2 rel err ~0.99) AND then fails the "Layout must be fully static"
  constraint on B. WORKAROUND that PASSES (L2 rel err 3.8e-3): call
  `multistage_gemm_q[group_size=32, pack_factor=8, config=cfg](c_lt, a_lt, b_lt,
  cfg, ctx)` directly (operands via `.to_layout_tensor()`) with a BK=32 config
  == the wrapper's own `default_config` (block 128x128x32, warp 64x64x32,
  stages=5, k_part=1, warp_k_part=1). This is the config the wrapper falls back
  to for non-tuned shapes; selecting it explicitly is the only working Q4_0 path.
- Consequence for the audit: on sm_86 the only WORKING Q4_0 matmul config at M=1
  is a 128x128 GEMM tile (no M=1 GEMV specialization — the m<=16 M16 config is
  the broken BK=128 one), so decode-shape Q4_0 runs badly underutilized. See
  bench/results and reports/open-questions.md.

## H0+ / our Q4_0 GEMV DP4A kernel (Mojo 1.0.0, ed45d567)

Facts verified against the installed stdlib while writing kernels/q4_0_gemv.mojo
(not drift per se, but non-obvious current API for GPU kernels):

- Shared memory: `unsafe_stack_allocation[count, T, address_space=AddressSpace.SHARED]()`
  from `std.memory`; `AddressSpace` is in `std.memory.address_space`.
- Block barrier: `from max.gpu.sync import barrier` (NOT in std.gpu). `std.gpu`
  has no `barrier`/`syncthreads`.
- Warp reductions: `std.gpu.primitives.warp` exposes `sum`, `max`, `min`,
  `broadcast`, `shuffle_*` taking `SIMD` values (e.g. `warp.max(abs(xf))`).
- Vector global loads: `ptr.unsafe_load[width=W, alignment=A](offset)` returns
  `SIMD[dtype, W]`; pass `alignment` explicitly for under-aligned data (Q4_0
  nibble bytes start at a 2-aligned offset -> `alignment=2` for a 16-byte load).
- SIMD reinterpret across widths: `from std.memory import bitcast`;
  `bitcast[DType.int32, 4](simd_u8x16)` repacks 16 uint8 lanes into 4 int32.
- No DP4A / dot-product intrinsic in the stdlib GPU package (only ARM neon
  `has_neon_int8_dotprod`). Emit it with inline PTX via
  `std.sys.inlined_assembly["dp4a.s32.s32 $0,$1,$2,$3;", Int32,
  constraints="=r,r,r,r", has_side_effect=False](a,b,c)`. This works on sm_86.
