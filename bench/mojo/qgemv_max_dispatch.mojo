# ===----------------------------------------------------------------------=== #
# MAX int4 (Q4_0) quantized matmul harness — REAL public dispatcher path.
#
# This harness measures what MAX ACTUALLY dispatches for each Llama-3-8B GEMV
# shape. It calls the PUBLIC wrapper `matmul_gpu_qint4[group_size=32]`, whose
# impl (matmul_gpu_qint4_impl, qmatmul_gpu.mojo:1842) selects a MatmulConfig
# from `static_K = a.layout.shape[1]` and `static_N = c.layout.shape[1]` at
# COMPILE TIME (cascade at :1896-2360). To let the real per-shape config be
# chosen, A and C carry STATIC K and N in their layouts.
#
# This CORRECTS bench/mojo/qgemv_max.mojo, which bypassed the dispatcher and
# forced multistage_gemm_q at block 128x128x32 (the wrapper's default_config)
# for ALL shapes. That was wrong: for shapes with a tuned cascade branch, MAX
# would pick a DIFFERENT config (e.g. BM=16, BK=32 for up/down_proj), and for
# o_proj/qkv the real tuned branch uses BK=128, which with group_size=32 gives
# group_size//BK == 0 -> a genuine compile failure (a real MAX g32 limitation).
#
# N and K are COMPILE-TIME here (top-of-file aliases), stamped per shape by the
# sweep driver, and one binary is built per shape so that shapes whose real
# config fails to compile (o_proj/qkv) do not block the shapes that do compile.
#
# Args (positional):  M N K W_path x_path ref_path [rtol] [atol]
#   M      : runtime batch (decode: 1); N,K in argv are asserted == comptime.
#   W_path : raw GGUF Q4_0 weight bytes, uint8, [N, (K//32)*18]
#   x_path : activations, bf16, [M, K]
#   ref_path: reference y, float32, [M, N]
# Output format matches qgemv_max.mojo so bench/run_gemv_max.py parses it:
#   device: / correctness: PASS|FAIL l2_rel_err= .. max_abs_err= .. max_rel_err=
#   launches_per_sample= 10 / samples_us= ...
# ===----------------------------------------------------------------------=== #

from quantization.qmatmul_gpu import gpu_qint4_repack_Q4_0, matmul_gpu_qint4
from layout import TileTensor, row_major, Coord, Idx
from max.gpu.host import DeviceContext
from std.memory import unsafe_memcpy
from std.sys import has_accelerator
from std.sys.arg import argv

# --- Per-shape compile-time dims. STAMPED by the sweep driver (sed). ---
comptime N = 14336
comptime K = 4096
# --- end stamp ---

comptime WARMUP = 10
comptime SAMPLES = 12
comptime PER_BATCH = 10
comptime GROUP_SIZE = 32
comptime GROUP_BYTES = 18  # Q4_0: 2 bytes fp16 scale + 32/2 packed 4-bit weights


def run(
    M: Int, w_path: String, x_path: String, ref_path: String,
    rtol: Float32, atol: Float32,
) raises:
    comptime assert K % GROUP_SIZE == 0, "K must be a multiple of the Q4_0 group size"

    comptime RAW_COLS = (K // GROUP_SIZE) * GROUP_BYTES  # per-row raw Q4_0 bytes
    comptime PACKED_COLS = (K * 9) // 16                  # per-row packed bytes
    comptime raw_bytes = N * RAW_COLS
    comptime packed_bytes = N * PACKED_COLS

    with DeviceContext() as ctx:
        print("device:", ctx.name())

        # ---- Host staging from raw .bin inputs (W uint8, x bf16, ref f32). ----
        var w_host = ctx.enqueue_create_host_buffer[DType.uint8](raw_bytes)
        var x_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
        var ref_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
        with open(w_path, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(dest=w_host.unsafe_ptr(),
                          src=raw.unsafe_ptr(),
                          count=raw_bytes)
        with open(x_path, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(dest=x_host.unsafe_ptr(),
                          src=raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.bfloat16]](),
                          count=M * K)
        with open(ref_path, "r") as f:
            var raw = f.read_bytes()
            unsafe_memcpy(dest=ref_host.unsafe_ptr(),
                          src=raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]](),
                          count=M * N)

        # ---- Device buffers; upload once (NOT timed). ----
        var w_dev = ctx.enqueue_create_buffer[DType.uint8](raw_bytes)
        var b_packed_dev = ctx.enqueue_create_buffer[DType.uint8](packed_bytes)
        var x_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
        var y_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
        ctx.enqueue_copy(w_dev, w_host)
        ctx.enqueue_copy(x_dev, x_host)
        b_packed_dev.enqueue_fill(0)
        y_dev.enqueue_fill(0)
        ctx.synchronize()

        # Repack views: N/K STATIC (kernel reads them at comptime).
        var b_raw_t = TileTensor(w_dev, row_major(Coord(Idx[N], Idx[RAW_COLS])))
        var b_packed_static_t = TileTensor(
            b_packed_dev, row_major(Coord(Idx[N], Idx[PACKED_COLS]))
        )

        # ---- Repack raw Q4_0 -> packed GEMM layout ONCE (not timed). ----
        gpu_qint4_repack_Q4_0[target="gpu"](b_raw_t, b_packed_static_t, ctx)
        ctx.synchronize()

        # Matmul operands with STATIC K on A and STATIC N on C so the dispatcher's
        # `static_K = a.layout.shape[1]` / `static_N = c.layout.shape[1]` pick the
        # REAL per-shape config. M stays runtime (dispatch reads m at launch).
        var a_t = TileTensor(x_dev, row_major(Coord(M, Idx[K])))
        var c_t = TileTensor(y_dev, row_major(Coord(M, Idx[N])))

        # ---- The op under test: the PUBLIC dispatcher. Routes through the
        #      real matmul_gpu_qint4_impl config cascade. ----
        @parameter
        @always_inline
        @__copy_capture(a_t, c_t, b_packed_static_t)
        def launch(ctx: DeviceContext) raises:
            matmul_gpu_qint4[group_size=GROUP_SIZE, target="gpu"](
                c_t, a_t, b_packed_static_t, ctx
            )

        # ---- Correctness: one launch, compare output (as fp32) to ref. ----
        launch(ctx)
        ctx.synchronize()
        var max_abs = Float32(0)
        var max_rel = Float32(0)
        var ssd = Float64(0)  # sum of squared (got - ref)
        var ssr = Float64(0)  # sum of squared ref
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


def main() raises:
    comptime assert has_accelerator(), "This harness requires a supported GPU"
    var args = argv()
    if len(args) < 7:
        print("usage: qgemv_max_dispatch M N K W_path x_path ref_path [rtol] [atol]")
        return
    var M = Int(String(args[1]))
    var argN = Int(String(args[2]))
    var argK = Int(String(args[3]))
    var w_path = String(args[4])
    var x_path = String(args[5])
    var ref_path = String(args[6])
    var rtol = Float32(atof(String(args[7]))) if len(args) > 7 else Float32(3e-2)
    var atol = Float32(atof(String(args[8]))) if len(args) > 8 else Float32(5e-3)

    if argN != N or argK != K:
        print("shape mismatch: this binary is stamped for N=", N, "K=", K,
              "but argv gave N=", argN, "K=", argK)
        return

    run(M, w_path, x_path, ref_path, rtol, atol)
