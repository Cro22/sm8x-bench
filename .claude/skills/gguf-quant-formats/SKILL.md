---
name: gguf-quant-formats
description: How GGUF quantized weight formats (Q8_0, Q4_0, Q4_K_M and friends) are laid out in memory, how to load real tensors from a .gguf file with gguf-py, how to produce the fp32 dequantized reference, and how to compute exact byte counts for roofline. Use whenever the task touches quantized weights, dequantization, GGUF files, block scales, bytes-per-weight, or comparing MAX quant formats against llama.cpp formats.
---

# GGUF quant formats

GGUF is the interchange format of local inference. Benchmarking against
llama.cpp means using its exact block layouts, so both implementations read the
same bytes. Layouts are defined authoritatively in llama.cpp's
`ggml/src/ggml-common.h` (structs) and `ggml/src/ggml-quants.c` (dequantize
functions). **When in doubt, read those files; do not trust this summary over
them.** `references/formats.md` has the byte-level summary for the three
formats we care about.

## Formats and priority

| Format | Block | Bytes/block | Bits/weight | Dequant | Use here |
|---|---|---|---|---|---|
| Q8_0 | 32 | 34 (fp16 d + 32×int8) | 8.5 | `d * q` | calibration format; simplest |
| Q4_0 | 32 | 18 (fp16 d + 16 bytes nibbles) | 4.5 | `d * (q - 8)` | simplest 4-bit; MAX-comparable |
| Q4_K (Q4_K_M) | 256 super-block | 144 (fp16 d, fp16 dmin, 12 B scales/mins 6-bit, 128 B nibbles) | 4.5 | `d*sc*q - dmin*m` per 32-sub-block | what people actually run |
| Q6_K, Q5_K, IQ4_XS | | | | | out of scope unless the audit says otherwise |

Note that "Q4_K_M" is a *model recipe* (mix of Q4_K, Q6_K for some tensors),
not a tensor format. When benchmarking a tensor, name the tensor format
(`Q4_K`); when describing a model file, say `Q4_K_M`.

## Loading real tensors

Use `gguf-py` (`uv add gguf`). `bench/gguf_load.py` should expose:

```python
load_tensor(path, name) -> (raw_bytes: np.uint8, ggml_type: str, shape: tuple)
dequantize(raw_bytes, ggml_type, shape) -> np.float32   # exact, matches ggml
```

`gguf-py` ships `gguf.quants.dequantize` — use it for the reference rather than
hand-writing dequant, and cross-check one block by hand once to confirm the
element ordering. The dequantized fp32 array is the reference for
`bench/reference.py`. Never quantize in Python and assume it matches ggml;
load blocks that llama-quantize produced.

Ask before downloading a model file (>1 GB). Prefer a small model with the same
tensor formats (e.g. a 1B Q4_K_M / Q8_0) for correctness, and synthetic random
blocks (valid layout, seeded) for the bandwidth sweep at Llama-3-8B shapes. Real
weights are needed for correctness because synthetic random nibbles never
exercise scale edge cases; synthetic is fine for timing because bandwidth does
not care about values.

## Layout gotchas that break correctness silently

- Q4_0 / Q4_K nibble order: byte `j` holds element `j` in the low nibble and
  element `j + half` in the high nibble (half = 16 for Q4_0; for Q4_K it is
  per 64-element group, see formats.md). Getting this wrong still produces a
  plausible-looking histogram of outputs — only the reference comparison
  catches it.
- Scales are fp16 (`ggml_half`), stored little-endian, at the *start* of the
  block for Q8_0/Q4_0.
- Q4_K 6-bit scale/min unpacking depends on the sub-block index (first four
  sub-blocks vs last four use different bit positions in `scales[12]`). Copy the
  logic from `ggml-quants.c` `get_scale_min_k4`, do not re-derive it.
- A GGUF tensor's `shape` is stored with the fastest-varying dimension first
  (ne[0] = K for a weight consumed as `y = W·x`). Row-major N×K in numpy means
  reversing that.
- Tensors in a Q4_K_M model are not all Q4_K. Check `ggml_type` per tensor.

## Bytes for roofline

`bench/roofline.py::bytes_per_weight(fmt)` returns `bytes_per_block /
block_size`: Q8_0 = 1.0625, Q4_0 = 0.5625, Q4_K = 0.5625. Fused-dequant
kernels read exactly these; a two-pass implementation (dequantize to fp16, then
GEMV) reads/writes 2 extra bytes per weight and should be reported as such,
with the traffic it actually incurs *and* the % of roofline it would achieve
against the minimum traffic. Both numbers, labeled.

## MAX-side formats

If the audit finds GPU quant formats in MAX that are not GGUF-compatible (e.g.
GPTQ/AWQ-style with per-channel or per-group scales, or a MAX-specific packed
layout), benchmark them at the same bits-per-weight and note that the byte
layouts differ; do not force a conversion that changes the traffic. Document
the mapping in `reports/audit.md`.

Read `references/formats.md` for byte-level structs.
