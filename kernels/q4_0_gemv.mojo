# ===----------------------------------------------------------------------=== #
# OUR OWN Q4_0 dequant-fused GEMV kernel (from scratch, not a MAX kernel).
#
# Computes  y[n] = sum_k dequant(W[n,k]) * x[k]   for n in 0..N-1
# with W stored as raw GGUF Q4_0 blocks (transpose_b / GEMV form), M == 1.
#
# GGUF Q4_0 layout: a row of K weights is K/32 blocks of 18 bytes each:
#   bytes[0:2]  = fp16 scale d (little-endian IEEE half)
#   bytes[2:18] = 16 nibble-bytes; weight j (0..15) = LOW nibble of byte[2+j],
#                 weight j+16 = HIGH nibble of byte[2+j].
#   dequant value = d * (nibble - 8).
# Row n starts at byte n*(K/32)*18 (row-major).
#
# ---------------------------------------------------------------------------
# Why the int8 / DP4A path (this is what llama.cpp's mul_mat_vec_q does):
# A pure fp32 dequant-and-FMA GEMV plateaus around 41% of the sm_86 memory
# roofline here (measured: ~28 us on o_proj). It is not DRAM-bandwidth bound at
# that point -- widening loads, adding ILP, staging x in smem, and sweeping the
# warps-per-CTA all left it at ~28-35 us. The limiter is the per-weight dequant
# work (int->float converts + FMAs, one chain per weight). llama.cpp sidesteps
# it by quantizing the activations to int8 (Q8_1) once and doing the Q4_0.Q8_1
# dot with DP4A (one instr = four int8 MACs), so the hot loop is a handful of
# integer instructions per block and the kernel becomes memory bound.
#
# Two kernels:
#   1) quantize_x_q8_1: x[K] (bf16) -> per-32-block int8 quants q8[K], plus
#      per-block scale xd[b] = amax/127 and xs[b] = xd[b]*sum(q8_block). Done
#      ONCE and reused across all N rows.
#   2) q4_0_gemv_kernel: each WARP computes ROWS_PER_WARP consecutive output
#      rows; each LANE owns whole Q4_0 blocks (block b handled by lane b%32,
#      stride 32). Per block it loads the 16 nibble-bytes as one 128-bit vector,
#      unpacks low/high nibbles into four int32 nibble-packs each (mask
#      0x0F0F0F0F), loads the 32 matching int8 x quants (8 int32) ONCE, and does
#      8 DP4A per row to get the integer dot D = sum_j q4_j * q8_j. The block's
#      contribution to row r is
#        d_w[r] * (xd[b]*D[r] - 8*xs[b])
#      since sum_j (q4_j-8)*(xd*q8_j) = xd*D - 8*xd*sum(q8) = xd*D - 8*xs.
#   Consecutive lanes read consecutive 18-byte blocks, so the warp sweeps a
#   contiguous region -> coalesced, every Q4_0 byte read once.
#
# Comptime knobs: ROWS_PER_WARP (rows/warp) and ROWS_PER_BLOCK (warps/CTA). More
# rows/warp gives each warp that many INDEPENDENT weight-load chains (the shared
# x-quants are loaded once), which is the memory-level parallelism that lifts the
# bandwidth-bound GEMV off its ~66% plateau; too many starves the small shapes of
# warps. Tuned per shape in the harness dispatch (bench/mojo/q4_0_gemv_ours.mojo):
# small/mid N want RPW=4; the two near-roofline giants want RPW=2, RPB=2.
#
# Measured (sm_86 RTX 3090, graphics clock 1695 MHz, nsys, incl. the quantize
# pass), M=1, per-shape tuned RPW/RPB. ours and llama.cpp mul_mat_vec_q on the SAME
# weights, measured SAME-SESSION, 3 passes each (least-contended min-pass %spec;
# the graphics-clock lock does not lock the GDDR6X memory clock so the run-to-run
# band is 0-9% and per-shape gaps <~5% are noise):
#   shape       ours min%   llama min%   verdict
#   o_proj        74.5        72.7        tie
#   qkv_fused     82.5        83.8        tie
#   up_proj       88.1        88.4        tie
#   down_proj     92.0        94.7        tie
#   gate_up       91.8       100.1        llama.cpp faster
#   lm_head      101.9       101.5        tie
# => PARITY with llama.cpp (no robust win for ours); the real win is vs MAX, which
# compile-fails o_proj/qkv (g32) and falls to a 15% GEMM tile on gate_up/lm_head.
# The o_proj/down_proj residual is the fixed ~2 us quantize-pass launch (same
# two-kernel structure llama.cpp uses); folding it into the GEMV was measured
# ~2x SLOWER (register-serial per-lane quantize doesn't hide), so it stays split.
# ===----------------------------------------------------------------------=== #

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.globals import WARP_SIZE
from std.gpu.primitives import warp
from std.memory import bitcast
from std.math import round
from std.sys import inlined_assembly

comptime GROUP_SIZE = 32
comptime GROUP_BYTES = 18  # 2 (fp16 scale) + 16 (nibble bytes)


@always_inline
def dp4a(a: Int32, b: Int32, c: Int32) -> Int32:
    # SM_61+ 4-way int8 dot-product-accumulate: c + sum_i a.i8[i]*b.i8[i].
    # Not exposed in the Mojo stdlib -> raw PTX. Signed x signed (q4 in 0..15
    # fits the positive range, q8 is signed -127..127).
    return inlined_assembly[
        "dp4a.s32.s32 $0, $1, $2, $3;",
        Int32,
        constraints="=r,r,r,r",
        has_side_effect=False,
    ](a, b, c)


def quantize_x_q8_1[
    K: Int
](
    q8: Pointer[Int8, MutAnyOrigin],         # [K] int8 quantized activations
    xd: Pointer[Float32, MutAnyOrigin],      # [K/32] per-block scale amax/127
    xs: Pointer[Float32, MutAnyOrigin],      # [K/32] per-block xd*sum(q8)
    x: Pointer[BFloat16, MutAnyOrigin],      # [K] bf16 activations
):
    # ONE WARP per 32-element block; lane l owns element l. GROUP_SIZE ==
    # WARP_SIZE, so amax/sum are single warp reductions (no serial 32-loop, and
    # the whole vector is quantized in parallel across the GPU).
    comptime NBLOCKS = K // GROUP_SIZE
    var b = (block_idx.x * block_dim.x + thread_idx.x) // WARP_SIZE
    if b >= NBLOCKS:
        return
    var lane = thread_idx.x % WARP_SIZE
    var base = b * GROUP_SIZE

    var xf = x.unsafe_load(base + lane).cast[DType.float32]()
    var amax = warp.max(abs(xf))
    var d = amax / 127.0
    var invd = (1.0 / d) if d > 0.0 else Float32(0.0)

    var qi = Int32(round(xf * invd))
    if qi > 127:
        qi = 127
    elif qi < -127:
        qi = -127
    q8.unsafe_store(base + lane, Int8(qi))
    var s = warp.sum(qi)
    if lane == 0:
        xd.unsafe_store(b, d)
        xs.unsafe_store(b, d * Float32(s))


def q4_0_gemv_kernel[
    N: Int, K: Int, ROWS_PER_BLOCK: Int, ROWS_PER_WARP: Int = 1
](
    y: Pointer[Float32, MutAnyOrigin],       # [N] fp32 output
    w: Pointer[UInt8, MutAnyOrigin],         # [N, (K/32)*18] raw Q4_0 bytes
    q8: Pointer[Int8, MutAnyOrigin],         # [K] int8 quantized activations
    xd: Pointer[Float32, MutAnyOrigin],      # [K/32] per-block x scale
    xs: Pointer[Float32, MutAnyOrigin],      # [K/32] per-block xd*sum(q8)
):
    # ONE WARP computes ROWS_PER_WARP consecutive output rows. The int8 x-quants
    # (q8/xd/xs) for a block are loaded ONCE and reused across all ROWS_PER_WARP
    # rows; each row keeps its own accumulator and its own weight-load stream, so
    # a warp has ROWS_PER_WARP independent DRAM-load chains in flight. That extra
    # memory-level parallelism is what lifts a bandwidth-bound GEMV that stalls at
    # ~66% of roofline (one load chain per warp) toward the roofline. N is a
    # multiple of ROWS_PER_WARP for every canonical shape -> no per-row tail.
    comptime assert N % ROWS_PER_WARP == 0, "N must be a multiple of ROWS_PER_WARP"
    comptime NBLOCKS = K // GROUP_SIZE            # Q4_0 blocks per row
    comptime ROW_BYTES = NBLOCKS * GROUP_BYTES    # raw bytes per weight row
    comptime FULL = NBLOCKS // WARP_SIZE          # whole 32-block warp sweeps

    var warp_id = thread_idx.x // WARP_SIZE
    var lane = thread_idx.x % WARP_SIZE
    var warp_global = block_idx.x * ROWS_PER_BLOCK + warp_id
    var n0 = warp_global * ROWS_PER_WARP          # first output row of this warp
    if n0 >= N:
        return

    var acc = InlineArray[Float32, ROWS_PER_WARP](fill=0.0)
    var mask = SIMD[DType.int32, 4](0x0F0F0F0F)

    @parameter
    @always_inline
    def process(b: Int):
        # x-quants for this 32-block: loaded ONCE, shared across the RPW rows.
        var qq = q8.unsafe_load[width=32, alignment=4](b * GROUP_SIZE)
        var qw = bitcast[DType.int32, 8](qq)
        var xdb = xd.unsafe_load(b)
        var xsb = xs.unsafe_load(b)
        var blk = b * GROUP_BYTES
        @parameter
        for r in range(ROWS_PER_WARP):
            var wblk = (n0 + r) * ROW_BYTES + blk
            var d_w = w.unsafe_bitcast[Scalar[DType.float16]]().unsafe_load(
                wblk // 2
            ).cast[DType.float32]()
            # 16 nibble-bytes -> four int32 words (128-bit vector load, 2-aligned).
            var qb = w.unsafe_load[width=16, alignment=2](wblk + 2)
            var words = bitcast[DType.int32, 4](qb)
            var lo = words & mask              # weights 0..15  (4 int8 per lane)
            var hi = (words >> 4) & mask       # weights 16..31
            # integer dot D = sum_j q4_j * q8_j via 8 DP4A.
            var dot: Int32 = 0
            @parameter
            for k in range(4):
                dot = dp4a(lo[k], qw[k], dot)
            @parameter
            for k in range(4):
                dot = dp4a(hi[k], qw[k + 4], dot)
            # contribution: d_w * (xd[b]*D - 8*xs[b])
            acc[r] += d_w * (xdb * Float32(dot) - 8.0 * xsb)

    for i in range(FULL):
        process(i * WARP_SIZE + lane)
    var tb = FULL * WARP_SIZE + lane
    if tb < NBLOCKS:
        process(tb)

    @parameter
    for r in range(ROWS_PER_WARP):
        var total = warp.sum(acc[r])
        if lane == 0:
            y.unsafe_store(n0 + r, total)
