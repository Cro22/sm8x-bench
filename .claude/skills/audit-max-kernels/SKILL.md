---
name: audit-max-kernels
description: How to inventory Modular's open-source kernel tree (modular/max/kernels and the Mojo stdlib gpu package) for LLM decode kernels — matmul/GEMV, quantized matmul, attention/MHA decode, paged KV cache — and determine what exists, which architectures it is gated to, how tile configs are selected, and whether it runs on sm_86/sm_89. Use whenever the task involves reading, mapping, or summarizing upstream MAX kernels, writing or updating reports/audit.md, deciding what kernel to benchmark or write next, or answering "does MAX already have X".
---

# Auditing `max/kernels`

The purpose of the audit is to answer one question with evidence: **where, if
anywhere, is upstream weak on consumer Ampere/Ada for LLM decode?** The answer
determines H1. Bias toward finding that upstream already does something well;
the failure mode to avoid is "wrote a kernel that already existed".

The submodule is pinned at `modular/` (read-only). Record its SHA in the report.

## Step 1 — locate the tree

Layout moves between releases. Start from the top:

```
ls modular/
find modular -maxdepth 3 -type d -iname "*kernel*"
find modular -type d -iname "*quant*" -o -type d -iname "*attention*" -o -type d -iname "*kv*cache*" | grep -v test
```

Expect something like `modular/max/kernels/src/{linalg,nn,quantization,kv_cache,...}`
and `modular/mojo/stdlib/std/gpu/`. Write the actual paths you find at the top
of the audit; do not assume.

## Step 2 — for each target area, fill the inventory

Targets (in priority order for decode):

1. **Quantized matmul / GEMV on GPU** — any Q4_0/Q8_0/Q4_K/GPTQ/AWQ-style weight
   format with GPU dispatch. Distinguish CPU-only implementations (historically
   the GGUF Q4 code was CPU) from GPU ones.
2. **Dense matmul at small M** (M=1..8, fp16/bf16 weights) — the GEMV path. Check
   whether there is a dedicated GEMV kernel or whether M=1 falls into the tiled
   GEMM.
3. **Attention decode** (MHA/GQA with KV from cache, single query token per
   sequence) — flash-decoding style split-K over sequence? Which KV dtypes?
4. **Paged KV cache** — block size, layout (which dims are contiguous), whether
   attention kernels read it directly.
5. **Supporting pieces:** RoPE, RMSNorm, SiLU/gate fusion, top-k/sampling —
   note only, do not benchmark.

For each kernel found, record in `reports/audit.md` (template in
`references/audit-template.md`):

- File path(s) and entry-point function name(s)
- Dtypes / quant formats supported
- **Arch gating:** grep for `sm_80|sm_86|sm_89|sm_90|sm_100|A100|H100|B200|
  MI300|MI355|has_nvidia_gpu_accelerator|is_nvidia|_is_sm|compute_capability|
  GPUInfo` around the kernel and its dispatch. Classify: (a) generic, no
  gating; (b) gated with a consumer-capable fallback; (c) gated to datacenter
  only (e.g. requires wgmma/TMA/FP8 paths with no sm_8x branch).
- **Config selection:** how tile sizes / block dims / num_stages are chosen.
  Look for lookup tables keyed by GPU name, autotune results, or hard-coded
  constants. If the table only has A100/H100/B200/MI3xx rows, note which row a
  4090 falls back to. This is where consumer under-tuning is most likely to
  live.
- **Tensor core usage:** which MMA path (mma.sync vs wgmma vs none).
- **Async copy / pipelining:** `cp.async`, `async_copy`, num_stages > 1.
- **Tests:** existing test files and whether they can be run on this box.
- **Runs on sm_86/sm_89?** Actually compile and run the smallest existing test
  or a minimal launch. `yes / yes with warnings / compile error (paste) /
  runtime error (paste) / not attempted (why)`.

## Step 3 — read the dispatch, not just the kernel

The interesting question is usually not "is there a kernel" but "what does MAX
actually launch for shape (1, 4096, 4096) Q4_0 on a 4090". Trace from the
public op (e.g. whatever `max.nn.Linear` or the graph `matmul` op lowers to)
down to the launch. Document the chain: op → dispatch function → config
selection → kernel. Use `nsys` on a tiny MAX graph if reading is ambiguous: the
kernel names in the trace are ground truth.

## Step 4 — write the gap analysis

End `reports/audit.md` with a section "Gaps and candidates for H1", ranked. A
gap is one of:

- missing entirely on GPU (e.g. no fused-dequant GEMV for a GGUF format)
- present but gated away from sm_8x
- present, runs, but config is a datacenter fallback (tile sizes assume 228 KB
  smem, 132 SMs, HBM3 latency, etc.)
- present and fine (say so — this is a valid and useful result)

For each real gap state: expected roofline headroom (from the H0 measurements,
once available), estimated effort, whether it needs tensor cores, and whether
the fix is a new kernel or a config table entry. A config-table PR is a
perfectly good H1 if that is what the data says.

## Conduct

- Quote upstream code by file path and line range, not by pasting large
  blocks. Keep the report readable by a Modular engineer who knows the tree.
- Where the code is confusing, write the confusion down as a question in
  `reports/open-questions.md` for the forum thread. Do not guess at intent.
- Do not modify anything under `modular/`. If you need to patch something to
  make a test run on this box, copy the file into `bench/mojo/patched/` and
  document the diff in the audit.
- Keep a running `reports/api-drift.md` of any Mojo API changes you had to adapt
  to (old → new). This is a byproduct with real value to Modular.
