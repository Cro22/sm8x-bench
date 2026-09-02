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
# Design (memory-bound GEMV): ONE WARP computes ONE output row n.
#   block = WARP_SIZE * ROWS_PER_BLOCK threads; grid = ceildiv(N, ROWS_PER_BLOCK).
#   Each warp strides over the K/32 blocks of its row. Within a block, lane l
#   (0..31) handles weight l: reads nibble-byte (l % 16), takes low nibble if
#   l < 16 else high nibble, dequantizes, multiplies by x[b*32 + l], accumulates
#   in fp32. After all blocks, warp.sum reduces the 32 lanes; lane 0 writes y[n].
#   Every Q4_0 weight byte is read exactly once (the goal).
# ===----------------------------------------------------------------------=== #

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.globals import WARP_SIZE
from std.gpu.primitives import warp

comptime GROUP_SIZE = 32
comptime GROUP_BYTES = 18  # 2 (fp16 scale) + 16 (nibble bytes)


def q4_0_gemv_kernel[
    N: Int, K: Int, ROWS_PER_BLOCK: Int
](
    y: Pointer[Float32, MutAnyOrigin],       # [N] fp32 output
    w: Pointer[UInt8, MutAnyOrigin],         # [N, (K/32)*18] raw Q4_0 bytes
    x: Pointer[BFloat16, MutAnyOrigin],      # [K] bf16 activations
):
    comptime NBLOCKS = K // GROUP_SIZE            # Q4_0 blocks per row
    comptime ROW_BYTES = NBLOCKS * GROUP_BYTES    # raw bytes per weight row

    var warp_id = thread_idx.x // WARP_SIZE
    var lane = thread_idx.x % WARP_SIZE
    var n = block_idx.x * ROWS_PER_BLOCK + warp_id
    if n >= N:
        return

    var row_base = n * ROW_BYTES
    var acc: Float32 = 0.0

    # Each lane handles weight index `lane` within every 32-weight block.
    var nib_byte = lane % 16              # which nibble-byte this lane reads
    var use_high = lane >= 16            # low nibble for l<16, high for l>=16

    for b in range(NBLOCKS):
        var blk = row_base + b * GROUP_BYTES
        # fp16 scale: 2 bytes little-endian at blk (blk is always even -> 2-aligned).
        # Index in float16 units on the bitcast pointer (blk is even).
        var d = w.unsafe_bitcast[Scalar[DType.float16]]().unsafe_load(
            blk // 2
        ).cast[DType.float32]()
        var byte = w.unsafe_load(blk + 2 + nib_byte)
        var nib = (byte >> 4) if use_high else (byte & UInt8(0x0F))
        var wv = d * (Float32(Int(nib)) - 8.0)
        var xv = x.unsafe_load(b * GROUP_SIZE + lane).cast[DType.float32]()
        acc += wv * xv

    var total = warp.sum(acc)
    if lane == 0:
        y.unsafe_store(n, total)
