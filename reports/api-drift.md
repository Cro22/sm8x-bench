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
