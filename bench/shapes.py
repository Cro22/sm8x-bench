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

WEIGHT_FORMATS = ["fp16", "Q8_0", "Q4_0", "Q4_K"]

# Decode attention: batch 1, one query token, KV from cache
ATTN_SEQ_LENS = [1024, 4096, 16384]
ATTN = dict(q_heads=Q_HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM, kv_dtype="fp16")
# paged KV block size: set from the MAX audit (reports/audit.md), do not guess
PAGE_SIZE = None

SEED = 20260901
