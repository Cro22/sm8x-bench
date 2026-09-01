# Open questions for forum.modular.com

Drafts only. Never posted by Claude Code.

## Internal (methodology, not for forum)

- **Decode-attention byte formula wording.** `bench-methodology` SKILL writes the
  minimum KV traffic as `seq_len * kv_heads * head_dim * 2 (K) * 2 (K and V) *
  bytes`, i.e. two factors of 2 (4x). The algorithmic minimum is one read of K
  plus one read of V = 2x. `bench/roofline.py:attention_decode_bytes` uses 2x.
  Confirm with Jesús that the skill wording is a typo, then fix the skill text.
  (Raised 2026-09-01.)
