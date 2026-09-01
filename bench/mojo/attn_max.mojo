# ===----------------------------------------------------------------------=== #
# MAX flash-decoding attention harness entry point (paged KV cache, decode).
#
# Launches the upstream MAX `nn.attention.gpu.mha.flash_attention` kernel over a
# PagedKVCacheCollection in the token-generation (decode) path: a single query
# token (valid_length=1) attends to `seq_len` cached KV positions. Validates the
# flash-decoding output against `mha_gpu_naive` (both reading the SAME paged KV
# cache), then times the flash_attention launch with the device-event timer.
#
# Config (fixed): batch_size=1, num_q_heads=32,
#   kv_params=KVCacheStaticParams(num_heads=8, head_size=128) -> GQA group 4,
#   page_size=128, num_layers=1, layer_idx=0, dtype fp16 (fallback bf16).
#
# Args (positional):  seq_len [rtol] [atol]
#   seq_len = KV context length (cached positions). rtol/atol default 2e-2/2e-3.
#
# Adapted from
#   modular/max/kernels/test/gpu/kv_cache/test_mha_decoding_vs_naive.mojo  and
#   modular/max/kernels/test/gpu/kv_cache/test_batch_kv_cache_flash_attention_causal_mask_ragged_paged.mojo
#
# Output format mirrors bench/mojo/gemv_max.mojo so the Python runner parses it:
#   device: <name>
#   correctness: PASS|FAIL l2_rel_err= <> max_abs_err= <> max_rel_err= <>
#   launches_per_sample= <PER_BATCH>
#   samples_us= v1,...,v12
# ===----------------------------------------------------------------------=== #

from std.math import ceildiv, rsqrt
from std.random import seed
from std.sys import has_accelerator
from std.sys.arg import argv
from std.utils import IndexList

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalMask

comptime WARMUP = 10
comptime SAMPLES = 12
comptime PER_BATCH = 10

comptime SEED = 20260901

# Fixed decode config.
comptime NUM_Q_HEADS = 32
comptime KV_PARAMS = KVCacheStaticParams(num_heads=8, head_size=128)
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime LAYER_IDX = 0


# Mirror of `padded_lut_cols` / `_LUT_TAIL_PAD` in
# modular/max/kernels/test/gpu/kv_cache/kv_cache_test_utils.mojo:
# PagedKVCache.populate's SIMD path needs the LUT row stride to be a multiple
# of 8 and at least cols+15. Production allocates with this padding.
def padded_lut_cols(cols: Int) -> Int:
    return ((cols + 7) // 8) * 8 + 16


def run[dtype: DType](seq_len: Int, rtol: Float32, atol: Float32) raises:
    with DeviceContext() as ctx:
        print("device:", ctx.name())

        comptime num_q_heads = NUM_Q_HEADS
        comptime kv_params = KV_PARAMS
        comptime page_size = PAGE_SIZE
        comptime head_size = kv_params.head_size

        var batch_size = 1
        var valid_length = 1  # decode: one new token
        var cache_length = seq_len  # cached KV positions
        var total_length = valid_length  # ragged Q rows (bs=1)
        var max_prompt_length = valid_length
        var max_full_context_length = cache_length + valid_length

        var num_pages = ceildiv(max_full_context_length, page_size)
        var num_paged_blocks = num_pages + 2  # a bit of slack

        # ---- Layouts (see paged flash-attention test). ----
        comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
        comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
        comptime q_layout = Layout.row_major(
            UNKNOWN_VALUE, num_q_heads, head_size
        )
        comptime kv_block_6d_layout = Layout.row_major[6]()
        comptime paged_lut_layout = Layout.row_major[2]()

        # ---- Row offsets [batch_size + 1]. ----
        var row_offsets = ManagedLayoutTensor[DType.uint32, row_offsets_layout](
            RuntimeLayout[row_offsets_layout].row_major(
                IndexList[1](batch_size + 1)
            ),
            ctx,
        )
        var row_offsets_host = row_offsets.tensor[update=False]()
        var running_offset: UInt32 = 0
        for i in range(batch_size):
            row_offsets_host[i] = running_offset
            running_offset += UInt32(valid_length)
        row_offsets_host[batch_size] = running_offset

        # ---- Cache lengths [batch_size]. ----
        var cache_lens = ManagedLayoutTensor[
            DType.uint32, cache_lengths_layout
        ](
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
            ctx,
        )
        var cache_lens_host = cache_lens.tensor[update=False]()
        for i in range(batch_size):
            cache_lens_host[i] = UInt32(cache_length)

        # ---- Ragged Q [total_length, num_q_heads, head_size], random. ----
        var q = ManagedLayoutTensor[dtype, q_layout](
            RuntimeLayout[q_layout].row_major(
                IndexList[3](total_length, num_q_heads, head_size)
            ),
            ctx,
        )
        random(q.tensor[update=False]())

        # ---- Output + naive reference [same shape as Q]. ----
        var output = ManagedLayoutTensor[dtype, q_layout](
            RuntimeLayout[q_layout].row_major(
                IndexList[3](total_length, num_q_heads, head_size)
            ),
            ctx,
        )
        var ref_output = ManagedLayoutTensor[dtype, q_layout](
            RuntimeLayout[q_layout].row_major(
                IndexList[3](total_length, num_q_heads, head_size)
            ),
            ctx,
        )

        # ---- Paged KV blocks
        # [num_paged_blocks, 2, num_layers, page_size, num_heads, head_size]. ----
        var kv_block_paged = ManagedLayoutTensor[dtype, kv_block_6d_layout](
            RuntimeLayout[kv_block_6d_layout].row_major(
                IndexList[6](
                    num_paged_blocks,
                    2,
                    NUM_LAYERS,
                    page_size,
                    kv_params.num_heads,
                    head_size,
                )
            ),
            ctx,
        )
        random(kv_block_paged.tensor[update=False]())

        # ---- Paged lookup table [batch_size, padded_lut_cols(num_pages)]. ----
        var paged_lut = ManagedLayoutTensor[DType.uint32, paged_lut_layout](
            RuntimeLayout[paged_lut_layout].row_major(
                IndexList[2](batch_size, padded_lut_cols(num_pages))
            ),
            ctx,
        )
        var paged_lut_host = paged_lut.tensor[update=False]()
        # Logical block i -> physical block i (batch_size == 1).
        for b in range(batch_size):
            for p in range(num_pages):
                paged_lut_host[b, p] = UInt32(p)

        # ---- Build the paged collection. ----
        var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
            kv_block_paged.device_tensor(),
            cache_lens.device_tensor(),
            paged_lut.device_tensor(),
            UInt32(max_prompt_length),
            UInt32(max_full_context_length),
        )

        var scale = rsqrt(Float32(head_size))

        var q_lt = q.device_tensor()
        var output_lt = output.device_tensor()
        var row_offsets_lt = row_offsets.device_tensor()

        @parameter
        @always_inline
        @__copy_capture(kv_collection, q_lt, output_lt, row_offsets_lt, scale)
        def launch(ctx: DeviceContext) raises:
            flash_attention[ragged=True](
                output_lt,
                q_lt,
                kv_collection.get_key_cache(LAYER_IDX),
                kv_collection.get_value_cache(LAYER_IDX),
                CausalMask(),
                row_offsets_lt,
                scale,
                ctx,
            )

        # ---- Flash-decoding: one launch. ----
        launch(ctx)
        ctx.synchronize()

        # ---- Naive reference over the same paged cache. ----
        mha_gpu_naive[ragged=True](
            q_lt,
            kv_collection.get_key_cache(LAYER_IDX),
            kv_collection.get_value_cache(LAYER_IDX),
            CausalMask(),
            ref_output.device_tensor(),
            row_offsets_lt,
            scale,
            batch_size,
            max_prompt_length,
            max_full_context_length,
            num_q_heads,
            head_size,
            num_q_heads // kv_params.num_heads,
            ctx,
        )
        ctx.synchronize()

        # ---- Correctness: relative L2 error over all [1,32,128] elements. ----
        var out_host = output.tensor()
        var ref_host = ref_output.tensor()
        var max_abs = Float32(0)
        var max_rel = Float32(0)
        var ssd = Float64(0)  # sum of squared (got - ref)
        var ssr = Float64(0)  # sum of squared ref
        for r in range(total_length):
            for h in range(num_q_heads):
                for d in range(head_size):
                    var got = out_host[r, h, d][0].cast[DType.float32]()
                    var rf = ref_host[r, h, d][0].cast[DType.float32]()
                    var ae = abs(got - rf)
                    var re = ae / (abs(rf) + Float32(1e-6))
                    if ae > max_abs:
                        max_abs = ae
                    if re > max_rel:
                        max_rel = re
                    var dd = Float64(got - rf)
                    ssd += dd * dd
                    ssr += Float64(rf) * Float64(rf)
        var l2_rel = Float32((ssd**0.5) / (ssr**0.5 + 1e-12))
        var ok = l2_rel < rtol
        print(
            "correctness:",
            "PASS" if ok else "FAIL",
            "l2_rel_err=",
            l2_rel,
            "max_abs_err=",
            max_abs,
            "max_rel_err=",
            max_rel,
        )
        if not ok:
            print("aborting: flash-decode L2 error", l2_rel, ">= tol", rtol)
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

        _ = row_offsets^
        _ = cache_lens^
        _ = q^
        _ = output^
        _ = ref_output^
        _ = kv_block_paged^
        _ = paged_lut^


def main() raises:
    comptime assert has_accelerator(), "This harness requires a supported GPU"
    var args = argv()
    if len(args) < 2:
        print("usage: attn_max seq_len [rtol] [atol]")
        return
    var seq_len = Int(String(args[1]))
    var rtol = Float32(atof(String(args[2]))) if len(args) > 2 else Float32(2e-2)
    var atol = Float32(atof(String(args[3]))) if len(args) > 3 else Float32(2e-3)

    seed(SEED)
    run[DType.float16](seq_len, rtol, atol)
