# Audit of `modular/max/kernels` for LLM decode on consumer NVIDIA GPUs

- Upstream SHA: `<sha>` (date)
- Mojo version: `<mojo --version>`
- Bench hardware: RTX 3090 (sm_86, driver X) [, RTX 4090 (sm_89, driver Y)]
- Tree paths found: `<actual paths>`

## Summary (5 lines max)

What exists, what runs on sm_8x, where the likely gap is.

## Inventory

### 1. Quantized matmul / GEMV (GPU)

| Kernel / entry point | Path | Formats | Arch gating | Config selection | MMA | Pipelining | Runs sm_86 | Runs sm_89 | Notes |
|---|---|---|---|---|---|---|---|---|---|

Dispatch chain for `(M=1, N=4096, K=4096, <format>)`: op → … → kernel.

### 2. Dense matmul at small M (fp16/bf16)

(same table)

### 3. Attention decode (MHA/GQA over KV cache)

(same table, plus: KV dtypes, split-K over sequence yes/no, GQA handling)

### 4. Paged KV cache

Block size, layout, which attention kernels consume it directly, dtypes.

### 5. Supporting kernels (note only)

RoPE, RMSNorm, activation fusions, sampling.

## Config tables and consumer fallback

Which GPU-keyed tables exist, which rows they have, what an RTX 3090 / RTX 4090
resolves to. Quote file:line.

## Compile / run log on this box

Per kernel attempted: command, outcome, paste of errors if any.

## Gaps and candidates for H1 (ranked)

For each: gap type, evidence (file:line and/or measurement), headroom estimate
(fill in after H0 measurements), effort, needs tensor cores?, fix type
(kernel / config / gating).

## Open questions for the forum

Moved to `reports/open-questions.md`; list titles here.
