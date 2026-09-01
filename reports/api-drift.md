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
- (More entries appended as harness entry points hit the compiler.)
