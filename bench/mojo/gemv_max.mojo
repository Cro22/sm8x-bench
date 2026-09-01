# ===----------------------------------------------------------------------=== #
# MAX matmul/GEMV harness entry point (generalized, dense fp16/bf16).
#
# Launches the upstream MAX `linalg.matmul.matmul` kernel for
# y[M,N] = x[M,K] @ W[N,K]^T (transpose_b). M=1 routes to the dedicated GEMV;
# M>1 bf16 -> the tiled tensor-core GEMM; M>1 fp16 -> cuBLAS (see audit).
# Validates against a precomputed fp32 reference, then times with the
# device-event timer.
#
# Args (all positional):  M N K fmt W_path x_path ref_path
#   fmt in {fp16, bf16} selects the weight+activation dtype (both share it).
# Inputs are raw little-endian, no header (see bench/reference.py).
#
# Methodology (.claude/skills/bench-methodology):
#   - Uploads once, OUTSIDE the timed region.
#   - Correctness (output cast to fp32 vs fp32 ref) BEFORE timing.
#   - Warmup >=10 untimed launches.
#   - Auto-calibrate LAUNCHES so one timed batch is >= ~5 ms (shape-independent).
#   - 30 samples; each times LAUNCHES back-to-back launches with
#     ctx.execution_time and divides by LAUNCHES. The Python runner parses the
#     printed `samples_us=` line and computes median/IQR.
# ===----------------------------------------------------------------------=== #

from linalg.matmul import matmul
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.memory import unsafe_memcpy
from std.sys import has_accelerator
from std.sys.arg import argv

comptime WARMUP = 10
# Fixed, modest launch budget: SAMPLES*PER_BATCH kernel instances — enough for
# stable nsys per-kernel stats, small enough that nsys report generation stays
# fast. nsys per-kernel duration is the authoritative timing; the wall-clock
# printed here is a secondary (dispatch-inclusive) sanity number.
comptime SAMPLES = 12
comptime PER_BATCH = 10


def run[dt: DType](
    M: Int, N: Int, K: Int, w_path: String, x_path: String, ref_path: String
) raises:
    with DeviceContext() as ctx:
        print("device:", ctx.name())

        # ---- Host staging, filled from raw .bin inputs (W,x in dt; ref f32). ----
        var w_host = ctx.enqueue_create_host_buffer[dt](N * K)
        var x_host = ctx.enqueue_create_host_buffer[dt](M * K)
        var ref_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
        with open(w_path, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(dest=w_host.unsafe_ptr(),
                          src=raw.unsafe_ptr().unsafe_bitcast[Scalar[dt]](),
                          count=N * K)
        with open(x_path, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(dest=x_host.unsafe_ptr(),
                          src=raw.unsafe_ptr().unsafe_bitcast[Scalar[dt]](),
                          count=M * K)
        with open(ref_path, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(dest=ref_host.unsafe_ptr(),
                          src=raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]](),
                          count=M * N)

        # ---- Device buffers; upload once (NOT timed). ----
        var w_dev = ctx.enqueue_create_buffer[dt](N * K)
        var x_dev = ctx.enqueue_create_buffer[dt](M * K)
        var y_dev = ctx.enqueue_create_buffer[dt](M * N)
        ctx.enqueue_copy(w_dev, w_host)
        ctx.enqueue_copy(x_dev, x_host)
        y_dev.enqueue_fill(0)
        ctx.synchronize()

        # a = x [M,K], b = W [N,K] (transpose_b), c = y [M,N].
        var a_t = TileTensor(x_dev, row_major(M, K))
        var b_t = TileTensor(w_dev, row_major(N, K))
        var c_t = TileTensor(y_dev, row_major(M, N))

        @parameter
        @always_inline
        @__copy_capture(a_t, b_t, c_t)
        def launch(ctx: DeviceContext) raises:
            matmul[transpose_b=True, target="gpu"](c_t, a_t, b_t, ctx)

        # ---- Correctness: one launch, compare output (as fp32) to ref. ----
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
                # rtol=1e-2, atol=1e-3 (dense fp16/bf16 tolerances).
                if ae > Float32(1e-3) + Float32(1e-2) * abs(r):
                    ok = False
                    if shown < 8:
                        print("  mismatch i=", i, " got=", got, " ref=", r)
                        shown += 1
        print("correctness:", "PASS" if ok else "FAIL",
              "max_abs_err=", max_abs, "max_rel_err=", max_rel)
        if not ok:
            print("aborting: kernel output does not match reference")
            return

        # ---- Warmup. ----
        for _ in range(WARMUP):
            launch(ctx)
        ctx.synchronize()

        # ---- Timed samples (fixed budget; wall-clock is secondary to nsys). ----
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


def main() raises:
    comptime assert has_accelerator(), "This harness requires a supported GPU"
    var args = argv()
    # args[0] is the program path.
    if len(args) < 8:
        print("usage: gemv_max M N K fmt W_path x_path ref_path")
        return
    var M = Int(String(args[1]))
    var N = Int(String(args[2]))
    var K = Int(String(args[3]))
    var fmt = String(args[4])
    var w_path = String(args[5])
    var x_path = String(args[6])
    var ref_path = String(args[7])

    if fmt == "fp16":
        run[DType.float16](M, N, K, w_path, x_path, ref_path)
    elif fmt == "bf16":
        run[DType.bfloat16](M, N, K, w_path, x_path, ref_path)
    else:
        print("unsupported fmt (dense path handles fp16|bf16):", fmt)
