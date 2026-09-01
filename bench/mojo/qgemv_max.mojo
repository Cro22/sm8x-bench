# ===----------------------------------------------------------------------=== #
# MAX int4 (Q4_0) quantized matmul/GEMV harness entry point.
#
# Launches the upstream MAX int4 quantized matmul for
#   y[M,N] = x[M,K] @ dequant(W)[N,K]^T   (transpose_b semantics)
# where W is stored in raw GGUF Q4_0 blocks. Two steps:
#   1) gpu_qint4_repack_Q4_0: repack the raw Q4_0 bytes into the packed GEMM
#      layout ONCE, before timing.
#   2) matmul_gpu_qint4: the quantized matmul; this is the op we time.
# Activations x and output y are bf16 (asserted by the kernel). Validates the
# output (cast to fp32) against a precomputed fp32 reference, then times with the
# device-event timer.
#
# Args (all positional):  M N K W_path x_path ref_path [rtol] [atol]
#   W_path : raw GGUF Q4_0 weight bytes, uint8, [N, (K//32)*18]
#   x_path : activations, bf16, [M, K]
#   ref_path: reference y, float32, [M, N] (= x_f32 @ dequant(W)_f32^T)
# Inputs are raw little-endian, no header (see bench/reference.py).
#
# N and K must be compile-time constants: the MAX repack kernel and the matmul
# dispatch both read N/K from the tile-tensor layout shape at comptime (grid
# geometry, packed layout, and the tuned MatmulConfig all depend on them). M
# stays a runtime value (the kernel derives M at launch). main() therefore
# dispatches the runtime (N,K) from argv to a static run[N,K] specialization
# covering the canonical Llama-3-8B GEMV shapes.
#
# Methodology (.claude/skills/bench-methodology): mirrors gemv_max.mojo.
#   - Upload + repack once, OUTSIDE the timed region.
#   - Correctness (output cast to fp32 vs fp32 ref) BEFORE timing.
#   - Warmup >=10 untimed launches; then SAMPLES batches of PER_BATCH launches
#     timed with ctx.execution_time; the Python runner parses `samples_us=`.
#   - Q4_0 is lossy 4-bit: default tolerance is looser (rtol 3e-2), gated on the
#     relative L2 error ||got-ref||/||ref||, not element-wise.
# ===----------------------------------------------------------------------=== #

from quantization.qmatmul_gpu import gpu_qint4_repack_Q4_0, multistage_gemm_q
from linalg.utils_gpu import MatmulConfig
from layout import TileTensor, row_major, Coord, Idx
from max.gpu.host import DeviceContext
from std.memory import unsafe_memcpy
from std.sys import has_accelerator
from std.sys.arg import argv
from std.utils.index import Index

comptime WARMUP = 10
comptime SAMPLES = 12
comptime PER_BATCH = 10
comptime GROUP_SIZE = 32
comptime PACK_FACTOR = 8
comptime GROUP_BYTES = 18  # Q4_0: 2 bytes fp16 scale + 32/2 packed 4-bit weights


def run[
    N: Int, K: Int
](
    M: Int, w_path: String, x_path: String, ref_path: String,
    rtol: Float32, atol: Float32,
) raises:
    comptime assert K % GROUP_SIZE == 0, "K must be a multiple of the Q4_0 group size"

    # Byte/element counts (see header + qmatmul_gpu.mojo repack layout).
    comptime RAW_COLS = (K // GROUP_SIZE) * GROUP_BYTES      # per-row raw Q4_0 bytes
    comptime PACKED_COLS = (K * 9) // 16                      # per-row packed bytes
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

        # Repack views: N/K STATIC. gpu_qint4_repack_Q4_0 reads N/K from the
        # layout at comptime (grid geometry + packed layout), so its operands
        # must carry static shapes.
        var b_raw_t = TileTensor(w_dev, row_major(Coord(Idx[N], Idx[RAW_COLS])))
        var b_packed_static_t = TileTensor(
            b_packed_dev, row_major(Coord(Idx[N], Idx[PACKED_COLS]))
        )

        # ---- Repack raw Q4_0 -> packed GEMM layout ONCE (not timed). ----
        gpu_qint4_repack_Q4_0[target="gpu"](b_raw_t, b_packed_static_t, ctx)
        ctx.synchronize()

        # Matmul operands. The quantized kernel (multistage_qgemm_kernel) derives
        # N and K at comptime from the *packed B* layout, so N and K must be
        # STATIC on all operands; M is read at runtime (c.dim[0]()), so M may be
        # dynamic. a = x [M,K] bf16, c = y [M,N] bf16, b = packed weights uint8.
        var a_t = TileTensor(x_dev, row_major(Coord(M, Idx[K])))
        var c_t = TileTensor(y_dev, row_major(Coord(M, Idx[N])))
        var a_lt = a_t.to_layout_tensor()
        var c_lt = c_t.to_layout_tensor()
        var b_lt = b_packed_static_t.to_layout_tensor()

        # Call MAX's multistage quantized GEMM directly with a BK=32 config
        # (block_tile[2] == Q4_0 group_size). We bypass the matmul_gpu_qint4
        # dispatch on purpose: for static 4096x4096 with group_size=32 it selects
        # BK=128 tuned configs where `group_size // BK == 0` produces a zero-sized
        # scales layout and a comptime failure (those configs were tuned for
        # group_size=128). This BK=32 config is exactly matmul_gpu_qint4's own
        # `default_config` fallback (block 128x128x32, warp 64x64x32).
        comptime cfg = MatmulConfig[
            DType.bfloat16, DType.uint8, DType.bfloat16, True
        ](
            block_tile_shape=Index(128, 128, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=5,
            num_k_partitions=1,
            num_warp_k_partitions=1,
        )

        @parameter
        @always_inline
        @__copy_capture(a_lt, b_lt, c_lt)
        def launch(ctx: DeviceContext) raises:
            multistage_gemm_q[
                group_size=GROUP_SIZE, pack_factor=PACK_FACTOR, config=cfg
            ](c_lt, a_lt, b_lt, cfg, ctx)

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
        # Gate on relative L2 error ||got-ref||/||ref||; rtol is the L2 tolerance.
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

        # ---- Timed samples (wall-clock is secondary to nsys). ----
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
    if len(args) < 7:
        print("usage: qgemv_max M N K W_path x_path ref_path [rtol] [atol]")
        return
    var M = Int(String(args[1]))
    var N = Int(String(args[2]))
    var K = Int(String(args[3]))
    var w_path = String(args[4])
    var x_path = String(args[5])
    var ref_path = String(args[6])
    # Q4_0 is lossy 4-bit: default to a looser tolerance than the dense harness.
    var rtol = Float32(atof(String(args[7]))) if len(args) > 7 else Float32(3e-2)
    var atol = Float32(atof(String(args[8]))) if len(args) > 8 else Float32(5e-3)

    # N and K must be comptime for the repack + matmul dispatch. Dispatch the
    # runtime (N,K) to a static specialization over the canonical GEMV shapes.
    if N == 4096 and K == 4096:
        run[4096, 4096](M, w_path, x_path, ref_path, rtol, atol)
    elif N == 6144 and K == 4096:
        run[6144, 4096](M, w_path, x_path, ref_path, rtol, atol)
    elif N == 4096 and K == 14336:
        run[4096, 14336](M, w_path, x_path, ref_path, rtol, atol)
    elif N == 14336 and K == 4096:
        run[14336, 4096](M, w_path, x_path, ref_path, rtol, atol)
    elif N == 28672 and K == 4096:
        run[28672, 4096](M, w_path, x_path, ref_path, rtol, atol)
    elif N == 128256 and K == 4096:
        run[128256, 4096](M, w_path, x_path, ref_path, rtol, atol)
    else:
        print("unsupported (N,K) for the quant harness:", N, K,
              "- add a static run[N,K] dispatch arm")
