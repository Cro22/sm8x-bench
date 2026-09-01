# ===----------------------------------------------------------------------=== #
# MAX matmul/GEMV harness entry point: decode-shaped M=1 fp16 GEMV.
#
# Launches the upstream MAX `linalg.matmul.matmul` kernel (which routes GPU M=1
# internally to its dedicated GEMV) for y[1,N] = x[1,K] @ W[N,K]^T, all fp16
# (fp32 accumulation inside), validates against a precomputed fp32 reference,
# then times it with the device-event timer.
#
# Shapes are the Llama-3-8B o_proj GEMV: N=4096, K=4096, M=1.
# Inputs are raw little-endian, no header, in bench/inputs/ (see task spec).
#
# Methodology (see .claude/skills/bench-methodology):
#   - Uploads happen once, OUTSIDE the timed region.
#   - Correctness (fp16 output cast to fp32 vs fp32 reference) BEFORE timing;
#     if it fails we print sample mismatches and stop.
#   - Warmup: >=10 untimed launches.
#   - 30 samples; each times LAUNCHES back-to-back launches with
#     ctx.execution_time and divides by LAUNCHES. The Python runner parses the
#     printed `samples_us=` line and computes median/IQR.
# ===----------------------------------------------------------------------=== #

from linalg.matmul import matmul
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.memory import unsafe_memcpy
from std.sys import has_accelerator

comptime M = 1
comptime N = 4096
comptime K = 4096

comptime W_PATH = "bench/inputs/gemv_o_proj_fp16_W.bin"  # [N, K] fp16, row-major
comptime X_PATH = "bench/inputs/gemv_o_proj_x.bin"  # [M, K] fp16
comptime REF_PATH = "bench/inputs/gemv_o_proj_fp16_ref.bin"  # [M, N] fp32

comptime RTOL = Float32(1e-2)
comptime ATOL = Float32(1e-3)

comptime WARMUP = 10
comptime SAMPLES = 30
comptime LAUNCHES = 256  # back-to-back launches per timed sample


def main() raises:
    comptime assert has_accelerator(), "This harness requires a supported GPU"

    with DeviceContext() as ctx:
        print("device:", ctx.name())

        # ---- Host staging buffers, filled from the raw .bin inputs. ----
        var w_host = ctx.enqueue_create_host_buffer[DType.float16](N * K)
        var x_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)

        with open(W_PATH, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(
                dest=w_host.unsafe_ptr(),
                src=raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.float16]](),
                count=N * K,
            )
        with open(X_PATH, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(
                dest=x_host.unsafe_ptr(),
                src=raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.float16]](),
                count=M * K,
            )

        # fp32 reference, kept on host for the correctness check.
        var ref_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
        with open(REF_PATH, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(
                dest=ref_host.unsafe_ptr(),
                src=raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]](),
                count=M * N,
            )

        # ---- Device buffers; upload once (NOT timed). ----
        var w_dev = ctx.enqueue_create_buffer[DType.float16](N * K)
        var x_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
        var y_dev = ctx.enqueue_create_buffer[DType.float16](M * N)
        ctx.enqueue_copy(w_dev, w_host)
        ctx.enqueue_copy(x_dev, x_host)
        y_dev.enqueue_fill(0)
        ctx.synchronize()

        # a = x [M, K], b = W [N, K] (transpose_b), c = y [M, N].
        var a_t = TileTensor(x_dev, row_major(M, K))
        var b_t = TileTensor(w_dev, row_major(N, K))
        var c_t = TileTensor(y_dev, row_major(M, N))

        @parameter
        @always_inline
        @__copy_capture(a_t, b_t, c_t)
        def launch(ctx: DeviceContext) raises:
            matmul[transpose_b=True, target="gpu"](c_t, a_t, b_t, ctx)

        # ---- Correctness: one launch, compare fp16 output (as fp32) to ref. ----
        launch(ctx)
        ctx.synchronize()

        var max_abs = Float32(0)
        var max_rel = Float32(0)
        var ok = True
        var shown = 0
        with y_dev.map_to_host() as h:
            for i in range(M * N):
                var got = h[i].cast[DType.float32]()
                var r = ref_host[i]
                var ae = abs(got - r)
                var re = ae / (abs(r) + Float32(1e-6))
                if ae > max_abs:
                    max_abs = ae
                if re > max_rel:
                    max_rel = re
                if ae > ATOL + RTOL * abs(r):
                    ok = False
                    if shown < 8:
                        print("  mismatch i=", i, " got=", got, " ref=", r)
                        shown += 1

        print(
            "correctness:",
            "PASS" if ok else "FAIL",
            "max_abs_err=",
            max_abs,
            "max_rel_err=",
            max_rel,
        )
        if not ok:
            print("aborting: kernel output does not match reference")
            return

        # ---- Warmup (untimed). ----
        for _ in range(WARMUP):
            launch(ctx)
        ctx.synchronize()

        # ---- 30 timed samples, each LAUNCHES launches on the device-event timer. ----
        var samples = List[Float64]()
        for _ in range(SAMPLES):
            var total_ns = Float64(ctx.execution_time[launch](LAUNCHES))
            var per_launch_us = total_ns / Float64(LAUNCHES) / 1000.0
            samples.append(per_launch_us)

        print("launches_per_sample=", LAUNCHES)
        var line = String("samples_us= ")
        for i in range(len(samples)):
            if i > 0:
                line += ","
            line += String(samples[i])
        print(line)
