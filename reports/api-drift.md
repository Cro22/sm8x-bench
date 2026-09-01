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
