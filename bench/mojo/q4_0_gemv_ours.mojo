# ===----------------------------------------------------------------------=== #
# Benchmark entry point for OUR OWN Q4_0 dequant-fused GEMV kernel.
#
# Launches kernels/q4_0_gemv.mojo's q4_0_gemv_kernel for
#   y[N] = sum_k dequant(W[n,k]) * x[k]   (transpose_b / GEMV form, M == 1)
# on raw GGUF Q4_0 weights, validates output (fp32) against the fp32 reference,
# then times with the device-event timer.
#
# Build (kernel lives in kernels/, add it to the import path):
#   uv run mojo build bench/mojo/q4_0_gemv_ours.mojo -I kernels \
#     -o bench/mojo/q4_0_gemv_ours
#
# Args (positional):  N K M W_path x_path ref_path [rtol] [atol]
#   W_path : raw GGUF Q4_0 weight bytes, uint8, [N, (K//32)*18]
#   x_path : activations, bf16, [1, K]
#   ref_path: reference y, float32, [1, N]  (= x_f32 @ dequant(W)_f32^T)
# ===----------------------------------------------------------------------=== #

from q4_0_gemv import q4_0_gemv_kernel, GROUP_SIZE, GROUP_BYTES
from std.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from std.memory import unsafe_memcpy
from std.sys import has_accelerator
from std.sys.arg import argv

comptime WARMUP = 10
comptime SAMPLES = 12
comptime PER_BATCH = 10
comptime ROWS_PER_BLOCK = 2  # best of {1,2,4,8,16} sweep on o_proj M=1


def median_of(list: List[Float64]) -> Float64:
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


def run[
    N: Int, K: Int
](
    w_path: String, x_path: String, ref_path: String,
    rtol: Float32, atol: Float32,
) raises:
    comptime assert K % GROUP_SIZE == 0, "K must be a multiple of the Q4_0 group size"
    comptime RAW_COLS = (K // GROUP_SIZE) * GROUP_BYTES
    comptime raw_bytes = N * RAW_COLS

    with DeviceContext() as ctx:
        print("device:", ctx.name())

        # ---- Host staging from raw .bin inputs. ----
        var w_host = ctx.enqueue_create_host_buffer[DType.uint8](raw_bytes)
        var x_host = ctx.enqueue_create_host_buffer[DType.bfloat16](K)
        var ref_host = ctx.enqueue_create_host_buffer[DType.float32](N)
        with open(w_path, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(dest=w_host.unsafe_ptr(),
                          src=raw.unsafe_ptr(),
                          count=raw_bytes)
        with open(x_path, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(dest=x_host.unsafe_ptr(),
                          src=raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.bfloat16]](),
                          count=K)
        with open(ref_path, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(dest=ref_host.unsafe_ptr(),
                          src=raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]](),
                          count=N)

        # ---- Device buffers; upload once (NOT timed). ----
        var w_dev = ctx.enqueue_create_buffer[DType.uint8](raw_bytes)
        var x_dev = ctx.enqueue_create_buffer[DType.bfloat16](K)
        var y_dev = ctx.enqueue_create_buffer[DType.float32](N)
        ctx.enqueue_copy(w_dev, w_host)
        ctx.enqueue_copy(x_dev, x_host)
        y_dev.enqueue_fill(0)
        ctx.synchronize()

        var w_ptr = w_dev.unsafe_ptr()
        var x_ptr = x_dev.unsafe_ptr()
        var y_ptr = y_dev.unsafe_ptr()

        comptime BLOCK = WARP_SIZE * ROWS_PER_BLOCK
        comptime GRID = (N + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

        @parameter
        @always_inline
        @__copy_capture(w_ptr, x_ptr, y_ptr)
        def launch(ctx: DeviceContext) raises:
            ctx.enqueue_function[
                q4_0_gemv_kernel[N, K, ROWS_PER_BLOCK]
            ](y_ptr, w_ptr, x_ptr, grid_dim=GRID, block_dim=BLOCK)

        # ---- Correctness: one launch, compare output to fp32 ref. ----
        launch(ctx)
        ctx.synchronize()
        var max_abs = Float32(0)
        var max_rel = Float32(0)
        var ssd = Float64(0)
        var ssr = Float64(0)
        with y_dev.map_to_host() as h:
            for i in range(N):
                var got = h[i]
                var r = ref_host[i]
                var ae = abs(got - r)
                var re = ae / (abs(r) + Float32(1e-6))
                if ae > max_abs:
                    max_abs = ae
                if re > max_rel:
                    max_rel = re
                var d = Float64(got - r)
                ssd += d * d
                ssr += Float64(r) * Float64(r)
        var l2_rel = Float32((ssd**0.5) / (ssr**0.5 + 1e-12))
        var ok = l2_rel < rtol
        print("correctness:", "PASS" if ok else "FAIL",
              "l2_rel_err=", l2_rel,
              "max_abs_err=", max_abs, "max_rel_err=", max_rel)
        if not ok:
            print("aborting: kernel output L2 error", l2_rel, ">= tol", rtol)
            with y_dev.map_to_host() as h:
                for i in range(min(N, 8)):
                    print("  got=", h[i], " ref=", ref_host[i])
            return

        # ---- Warmup. ----
        for _ in range(WARMUP):
            launch(ctx)
        ctx.synchronize()

        # ---- Timed samples. ----
        var samples = List[Float64]()
        for _ in range(SAMPLES):
            var total_ns = Float64(ctx.execution_time[launch](PER_BATCH))
            samples.append((total_ns / Float64(PER_BATCH)) / 1000.0)

        print("launches_per_sample=", PER_BATCH)
        var line = String("samples_us= ")
        for i in range(len(samples)):
            if i > 0:
                line += ","
            line += String(samples[i])
        print(line)
        print("median_us=", median_of(samples), " rows_per_block=", ROWS_PER_BLOCK)


def main() raises:
    comptime assert has_accelerator(), "This harness requires a supported GPU"
    var args = argv()
    if len(args) < 7:
        print("usage: q4_0_gemv_ours N K M W_path x_path ref_path [rtol] [atol]")
        return
    var N = Int(String(args[1]))
    var K = Int(String(args[2]))
    var M = Int(String(args[3]))
    var w_path = String(args[4])
    var x_path = String(args[5])
    var ref_path = String(args[6])
    var rtol = Float32(atof(String(args[7]))) if len(args) > 7 else Float32(3e-2)
    var atol = Float32(atof(String(args[8]))) if len(args) > 8 else Float32(5e-3)

    if M != 1:
        print("this GEMV kernel supports M==1 only for now; got M=", M)
        return

    if N == 4096 and K == 4096:
        run[4096, 4096](w_path, x_path, ref_path, rtol, atol)
    elif N == 6144 and K == 4096:
        run[6144, 4096](w_path, x_path, ref_path, rtol, atol)
    elif N == 4096 and K == 14336:
        run[4096, 14336](w_path, x_path, ref_path, rtol, atol)
    elif N == 14336 and K == 4096:
        run[14336, 4096](w_path, x_path, ref_path, rtol, atol)
    elif N == 28672 and K == 4096:
        run[28672, 4096](w_path, x_path, ref_path, rtol, atol)
    elif N == 128256 and K == 4096:
        run[128256, 4096](w_path, x_path, ref_path, rtol, atol)
    else:
        print("unsupported (N,K):", N, K, "- add a static run[N,K] dispatch arm")
