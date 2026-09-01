# Byte-level layouts (summary — authoritative source is ggml-common.h / ggml-quants.c)

All multi-byte fields little-endian. `half` = IEEE fp16.

## Q8_0 — block of 32

```
struct block_q8_0 {           // 34 bytes
    half   d;                 // scale
    int8_t qs[32];            // quants
};
value[i] = d * qs[i]
```

## Q4_0 — block of 32

```
struct block_q4_0 {           // 18 bytes
    half    d;                // scale
    uint8_t qs[16];           // 32 nibbles
};
value[j]      = d * ((qs[j] & 0x0F) - 8)     for j in 0..15
value[j + 16] = d * ((qs[j] >> 4)   - 8)     for j in 0..15
```

## Q4_K — super-block of 256 (8 sub-blocks of 32)

```
struct block_q4_K {           // 144 bytes  (QK_K = 256)
    half    d;                // super-block scale for scales
    half    dmin;             // super-block scale for mins
    uint8_t scales[12];       // 8 × 6-bit scale + 8 × 6-bit min, packed
    uint8_t qs[128];          // 256 nibbles
};
```

Dequant, following `dequantize_row_q4_K`:

```
for each of 4 groups g (64 elements each, 32 bytes of qs):
    sc0, m0 = get_scale_min_k4(2g,   scales)
    sc1, m1 = get_scale_min_k4(2g+1, scales)
    d1 = d * sc0;  min1 = dmin * m0
    d2 = d * sc1;  min2 = dmin * m1
    for l in 0..31:  out[64g + l]      = d1 * (qs[32g + l] & 0x0F) - min1
    for l in 0..31:  out[64g + 32 + l] = d2 * (qs[32g + l] >> 4)   - min2
```

`get_scale_min_k4(j, q)`:

```
if j < 4:  sc = q[j] & 63;                       m = q[j+4] & 63
else:      sc = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
           m  = (q[j+4] >>  4)  | ((q[j]   >> 6) << 4)
```

Copy this from the C source when implementing; the bit positions are easy to
transpose.

## Bytes per weight

| Format | bytes/block | block | bytes/weight |
|---|---|---|---|
| fp16 | – | – | 2.0 |
| Q8_0 | 34 | 32 | 1.0625 |
| Q4_0 | 18 | 32 | 0.5625 |
| Q4_K | 144 | 256 | 0.5625 |
| Q6_K | 210 | 256 | 0.8203 |

## Row layout of a weight tensor

A GGUF weight consumed as `y = W·x` with W logically N×K is stored row by row:
row n = `K / block_size` consecutive blocks. Blocks never straddle rows (K is a
multiple of 32 for all Llama shapes; 256 for Q4_K — 4096, 14336 and 128256 are
all multiples of 256).
