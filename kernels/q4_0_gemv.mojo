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
#   2) q4_0_gemv_kernel: ONE WARP per output row; each LANE owns whole Q4_0
#      blocks (block b handled by lane b%32, stride 32). Per block it loads the
#      16 nibble-bytes as one 128-bit vector, unpacks low/high nibbles into four
#      int32 nibble-packs each (mask 0x0F0F0F0F), loads the 32 matching int8 x
#      quants (8 int32), and does 8 DP4A to get the integer dot
#      D = sum_j q4_j * q8_j. The block's contribution is
#        d_w * (xd[b]*D - 8*xs[b])
#      since sum_j (q4_j-8)*(xd*q8_j) = xd*D - 8*xd*sum(q8) = xd*D - 8*xs.
#   Consecutive lanes read consecutive 18-byte blocks, so the warp sweeps a
#   contiguous region -> coalesced, every Q4_0 byte read once.
#
# Comptime knobs (retune later): ROWS_PER_BLOCK (warps per CTA). Best on sm_86
# (RTX 3090) is ROWS_PER_BLOCK=1 across the canonical shapes (lm_head prefers it
# clearly; o_proj is within noise of 2). See the sweep in reports.
#
# Measured (sm_86, locked 1695 MHz, nsys, incl. the quantize pass), M=1:
#   shape     median us   % spec roofline   (fp32-dequant baseline -> now)
#   o_proj      16.3         ~62%           (35.8% -> 62%)
#   up_proj     43.1         ~82%           (39.9% -> 82%)
#   lm_head    337           ~94%           (43.2% -> 94%)
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
    N: Int, K: Int, ROWS_PER_BLOCK: Int
](
    y: Pointer[Float32, MutAnyOrigin],       # [N] fp32 output
    w: Pointer[UInt8, MutAnyOrigin],         # [N, (K/32)*18] raw Q4_0 bytes
    q8: Pointer[Int8, MutAnyOrigin],         # [K] int8 quantized activations
    xd: Pointer[Float32, MutAnyOrigin],      # [K/32] per-block x scale
    xs: Pointer[Float32, MutAnyOrigin],      # [K/32] per-block xd*sum(q8)
):
    comptime NBLOCKS = K // GROUP_SIZE            # Q4_0 blocks per row
    comptime ROW_BYTES = NBLOCKS * GROUP_BYTES    # raw bytes per weight row
    comptime FULL = NBLOCKS // WARP_SIZE          # whole 32-block warp sweeps

    var warp_id = thread_idx.x // WARP_SIZE
    var lane = thread_idx.x % WARP_SIZE
    var n = block_idx.x * ROWS_PER_BLOCK + warp_id
    if n >= N:
        return

    var row_base = n * ROW_BYTES
    var acc: Float32 = 0.0

    @parameter
    @always_inline
    def process(b: Int):
        var blk = row_base + b * GROUP_BYTES
        var d_w = w.unsafe_bitcast[Scalar[DType.float16]]().unsafe_load(
            blk // 2
        ).cast[DType.float32]()
        # 16 nibble-bytes -> four int32 words (128-bit vector load, 2-aligned).
        var qb = w.unsafe_load[width=16, alignment=2](blk + 2)
        var words = bitcast[DType.int32, 4](qb)
        var mask = SIMD[DType.int32, 4](0x0F0F0F0F)
        var lo = words & mask              # weights 0..15  (4 int8 per lane)
        var hi = (words >> 4) & mask       # weights 16..31
        # 32 matching int8 x-quants -> eight int32.
        var qq = q8.unsafe_load[width=32, alignment=4](b * GROUP_SIZE)
        var qw = bitcast[DType.int32, 8](qq)
        # integer dot D = sum_j q4_j * q8_j via 8 DP4A.
        var dot: Int32 = 0
        @parameter
        for k in range(4):
            dot = dp4a(lo[k], qw[k], dot)
        @parameter
        for k in range(4):
            dot = dp4a(hi[k], qw[k + 4], dot)
        # contribution: d_w * (xd[b]*D - 8*xs[b])
        acc += d_w * (xd.unsafe_load(b) * Float32(dot) - 8.0 * xs.unsafe_load(b))

    for i in range(FULL):
        process(i * WARP_SIZE + lane)
    var tb = FULL * WARP_SIZE + lane
    if tb < NBLOCKS:
        process(tb)

    var total = warp.sum(acc)
    if lane == 0:
        y.unsafe_store(n, total)
