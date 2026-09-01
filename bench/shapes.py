"""Canonical shapes. Single source of truth — every harness, baseline, and
report imports from here. Llama-3-8B decode."""

HIDDEN = 4096
INTERMEDIATE = 14336
VOCAB = 128256
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128

# (name, N, K) for y = W @ x with W stored N x K
GEMV_SHAPES = [
    ("o_proj",      HIDDEN,           HIDDEN),
    ("qkv_fused",   HIDDEN + 2 * KV_HEADS * HEAD_DIM, HIDDEN),   # 6144 x 4096
    ("down_proj",   HIDDEN,           INTERMEDIATE),
    ("up_proj",     INTERMEDIATE,     HIDDEN),
    ("gate_up",     2 * INTERMEDIATE, HIDDEN),                   # 28672 x 4096
    ("lm_head",     VOCAB,            HIDDEN),
]

M_VALUES = [1, 8]

# Union of formats across implementations. Per-impl support (from the audit):
#   MAX GPU: fp16, bf16, Q4_0   (no GPU Q8_0/Q4_K; fp16 M>1 -> cuBLAS)
#   llama.cpp: Q8_0, Q4_0, Q4_K_M (+ fp16 via cuBLAS reference)
# reports mark unsupported (impl, format) cells as N/A.
WEIGHT_FORMATS = ["fp16", "bf16", "Q8_0", "Q4_0", "Q4_K"]

# Decode attention: batch 1, one query token, KV from cache
ATTN_SEQ_LENS = [1024, 4096, 16384]
ATTN = dict(q_heads=Q_HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM, kv_dtype="fp16")
# paged KV block size: from the MAX audit (reports/audit.md). MAX ships 128 as
# the default AND enforces page_size % 128 == 0 and >= 128
# (max/python/max/pipelines/kv_cache/registry.py:50). So 128 is the only value
# a Llama-3-8B decode run uses unless explicitly overridden.
PAGE_SIZE = 128

SEED = 20260901
