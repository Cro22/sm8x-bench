# ===----------------------------------------------------------------------=== #
# GPU memory-bandwidth probe (measured streaming roofline).
#
# Measures the sustained DRAM read bandwidth of the attached GPU by launching a
# pure streaming READ reduction over a >= 2 GiB device buffer. The result is the
# "measured memory roofline" we compare kernels against (spec for the RTX 3090
# is 936 GB/s).
#
# Methodology (see .claude/skills/bench-methodology):
#   - Allocate N int32 (2 GiB), fill on device (allocation/upload NOT timed).
#   - Kernel: grid-stride vectorized read of the whole buffer, warp-reduce, one
#     atomic add per warp into a 1-element output. Every byte is read once.
#   - Warmup: 10 untimed launches.
#   - Sample: 30 measurements; each times K back-to-back launches (K chosen so a
#     batch is >= 5 ms) and divides by K. Timing uses the device-event timer
#     (ctx.execution_time), i.e. CUDA events, not host wall-clock.
#   - Report best-of-30 (closest to true roofline) and median GB/s.
# Correctness: fill with 1 -> sum must equal N exactly (fits in int32).
# ===----------------------------------------------------------------------=== #

from std.gpu import global_idx, thread_idx, block_dim, grid_dim
from std.gpu.globals import WARP_SIZE
from std.gpu.primitives import warp
from std.atomic import Atomic
from max.gpu.host import DeviceContext
from std.sys import has_accelerator, size_of

comptime dtype = DType.int32
comptime N = 1 << 29  # 536,870,912 elements * 4 B = 2 GiB
comptime SIMD_W = 4  # vectorized load width for coalesced DRAM reads
comptime NVEC = N // SIMD_W
comptime BLOCK = 256
comptime GRID = 4096  # 1,048,576 threads, each grid-strides over the buffer
comptime BYTES = N * size_of[dtype]()

comptime WARMUP = 10
comptime SAMPLES = 30
comptime K = 4  # launches per timed batch (~2 GiB read ~2.7 ms -> ~11 ms/batch)


def bw_kernel(
    output: Pointer[Int32, MutAnyOrigin], a: Pointer[Int32, MutAnyOrigin]
):
    """Streaming read reduction: sum every element of `a` into `output[0]`."""
    var gid = global_idx.x
    var stride = grid_dim.x * block_dim.x
    var local: Int32 = 0
    # Grid-stride over the buffer in SIMD_W-wide vectorized loads.
    for i in range(gid, NVEC, stride):
        local += a.unsafe_load[width=SIMD_W](i * SIMD_W).reduce_add()
    # One atomic per warp: reduce across the warp, then lane 0 accumulates.
    var wsum = warp.sum(local)
    if (thread_idx.x % WARP_SIZE) == 0:
        _ = Atomic.fetch_add(output, wsum)


def median_of(list: List[Float64]) -> Float64:
    # Copy + insertion sort (SAMPLES is tiny); return the median.
    var v = list.copy()
    for i in range(1, len(v)):
        var key = v[i]
        var j = i - 1
        while j >= 0 and v[j] > key:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = key
    var n = len(v)
    if n % 2 == 1:
        return v[n // 2]
    return (v[n // 2 - 1] + v[n // 2]) / 2.0


def main() raises:
    comptime assert has_accelerator(), "This probe requires a supported GPU"

    with DeviceContext() as ctx:
        print("device:", ctx.name())

        var out = ctx.enqueue_create_buffer[dtype](1)
        var a = ctx.enqueue_create_buffer[dtype](N)
        a.enqueue_fill(1)  # every element == 1 -> sum == N
        ctx.synchronize()

        @parameter
        @always_inline
        @__copy_capture(out, a)
        def launch(ctx: DeviceContext) raises:
            ctx.enqueue_function[bw_kernel](
                out, a, grid_dim=GRID, block_dim=BLOCK
            )

        # ---- Correctness: single launch, verify sum == N exactly. ----
        out.enqueue_fill(0)
        launch(ctx)
        ctx.synchronize()
        var got: Int
        with out.map_to_host() as h:
            got = Int(h[0])
        var ok = got == N
        print(
            "correctness:",
            "PASS" if ok else "FAIL",
            "sum=",
            got,
            "expected=",
            N,
        )
        if not ok:
            print("aborting: kernel did not read the whole buffer correctly")
            return

        # ---- Warmup (untimed). ----
        for _ in range(WARMUP):
            launch(ctx)
        ctx.synchronize()

        # ---- 30 timed samples, each K launches on the device-event timer. ----
        var gbps = List[Float64]()
        var best: Float64 = 0.0
        for _ in range(SAMPLES):
            var total_ns = Float64(ctx.execution_time[launch](K))
            var per_launch_ns = total_ns / Float64(K)
            # bytes / ns == GB/s (1 byte/ns = 1e9 B/s = 1 GB/s).
            var g = Float64(BYTES) / per_launch_ns
            gbps.append(g)
            if g > best:
                best = g

        var med = median_of(gbps)
        print(
            "measured_bw_gbps_best=",
            best,
            " median=",
            med,
            " N_bytes=",
            BYTES,
            " K=",
            K,
        )
