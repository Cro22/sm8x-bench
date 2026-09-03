# Plan: fixing MAX's Q4_0 / decode failures upstream

What H0 found in `modular/max/kernels` (pinned tag `max/v26.5.0`, SHA `b4497b7`)
on consumer Ampere (RTX 3090, sm_86), and how each would be fixed and upstreamed.
Every claim here is backed by a measured result in `bench/results/` or a verified
source line; the compile failure was reproduced this session (see
`bench/results/max_gemv_Q4-0-M1-*.compile_error.txt`).

## Upstream state — checked before proposing anything (2026-09-03)

`modular/main` (many months ahead of the pinned `v26.5.0`) still has **both** bugs
— the line numbers only shifted: `group_size // BK` is unguarded at
`qmatmul_gpu.mojo:456,518`, the 4096×4096 cascade still picks `BK=128` for
`m≤16`/`m≤32` with no group_size guard (`:1417,1429`), and the A-tile load
`a.tiled_iterator[BM, BK]` (`:737`) is still unmasked in the shared-memory prefetch.

**Two open, directly-relevant PRs — a chain by which Modular is already bringing
decode/serving quantization onto the multistage skeleton. RULE: we do not touch
anything that already has a PR.** Searched the open-PR list per issue (GitHub
search matches title+body, so "none found" ≠ absolute):

> **[modular#6668](https://github.com/modular/modular/pull/6668) (open)** —
> "[Kernels][Pipelines] NVFP4 fallback for pre-Blackwell NVIDIA GPUs via **fused
> dequant-GEMV**". A batch-1, bandwidth-bound decode GEMV ("per-token cost is
> essentially reading the packed weights once") for **NVFP4**, on pre-Blackwell
> (incl. sm_86).
>
> **[modular#6708](https://github.com/modular/modular/pull/6708) (open, DRAFT)** —
> "[Kernels][GPU] NVFP4 GEMM on the multistage qGEMM skeleton", builds on #6668.
> Reworks `multistage_mma_q` and **fixes `group_size < BK`** (*"Fixed
> num_scales_stages for group < BK … stale scale at group=16"*), validating
> **group=32**.

Per-issue PR status (verified 2026-09-03):

| Issue | Open PR? | What we do |
|---|---|---|
| **A. Q4_0 g32 compile-fail** | **#6708** fixes group<BK, validates g32 (draft) | **Don't touch.** Only, once it lands: verify GGUF Q4_0 on sm_86 + add the Q4_0 test it lacks. |
| **B. Unmasked A-tile load** | **FILED as [#7069](https://github.com/modular/modular/issues/7069)** (no prior issue/PR) | Writeup `max-B-fix.md`; PR next (needs a MAX build env). |
| **C. No M=1 decode-quant GEMV** | **#6668** does exactly this **for NVFP4** | **Don't build a competing GEMV.** The pattern is already upstream. Only verify whether Q4_0 rides it; if not, that wiring is the niche. |
| **D. Attention seq-4096 gap** | **None** (and not a confirmed defect) | Just the paged-FlashInfer measurement to confirm/deny. |

So under "don't duplicate a PR", the actionable work shrinks to: **B** (real fix),
**D's measurement**, and **Q4_0-specific tests/verification** riding on the #6668/
#6708 chain once it lands. A and C are *track-only*.

## Ground rules

- The `modular/` submodule is **read-only** here. Fixes are drafted in this doc,
  implemented in a **separate clone** of `modular/main`, and sent as PRs to
  `modular/max/kernels`. Nothing in this repo edits `modular/`.
- Rebase each fix onto `modular/main` before submitting (the pinned SHA is old;
  line numbers below are for `max/v26.5.0` and must be re-confirmed on `main`).
- Commit style (from `modular/max/kernels/CLAUDE.md`): `[Kernel]` / `[GPU]` tags,
  body wrapped in `BEGIN_PUBLIC` / `END_PUBLIC`, signed. Each fix ships with a test.
- Land the two **bug fixes** (A, B) first — small, self-contained, obviously
  correct. The **coverage gap** (C) is the real perf contribution and is larger.
  The **attention gap** (D) is not yet a confirmed MAX defect — it needs one more
  measurement before any fix is proposed.

---

## A. Q4_0 (group_size=32) fails to compile on the M≤32 tuned configs — BUG

**Symptom (measured).** `matmul_gpu_qint4[group_size=32]` — MAX's *public*
quantized-matmul entry — fails to compile for the static shapes whose tuned
config is selected for `m ≤ 32`, i.e. o_proj (N=K=4096) and qkv (N=6144, K=4096).
Reproduced through the real dispatcher; full compiler output committed as
`bench/results/max_gemv_Q4-0-M1-{o_proj,qkv_fused}.compile_error.txt`. up_proj /
down_proj compile and run (they hit a BK=32 config).

**Root cause (verified, file:line in v26.5.0).**
`quantization/qmatmul_gpu.mojo:1896+` (`matmul_gpu_qint4_impl` config cascade)
selects, for 4096×4096, `block_tile_shape=Index(16, 64, 128)` (`m≤16`, line 1901)
or `Index(32, 64, 128)` (`16<m≤32`, line 1924) — **BK = 128**. The multistage
kernel then assumes each scale group spans `group_size // BK` BK-blocks:
`qmatmul_gpu.mojo:374` `if stage % (group_size // BK) == 0:` (also lines 529, 602).
For GGUF Q4_0, `group_size = 32`, so `group_size // BK = 32 // 128 = 0` → `stage %
0` / a zero-sized scales dimension → the comptime `int_tuple` out-of-bounds. The
tuned table implicitly assumes `group_size ≥ BK` (a GPTQ/AWQ g128 assumption); g32
is second-class. This is upstream open-question #5 in `reports/open-questions.md`.

**Fix — NOTE: largely being handled by [modular#6708](https://github.com/modular/modular/pull/6708)**
(see "Upstream state" above; it fixes group < BK and validates group=32). Track it
rather than duplicate. Our value-add is the **Q4_0 regression test** it lacks and
**sm_86 verification**. The options below are the fallback if #6708 stalls.
- **A1 (cheap, safe, standalone stopgap): guard the config choice by group size.**
  In the cascade, gate every `BK=128` config on `group_size >= 128` (or
  `group_size % BK == 0`); when the guard fails, fall through to a `BK=32` config
  (up_proj/down_proj already define such configs) or `default_config`. Effect:
  o_proj/qkv **compile and run** for Q4_0 — at the BK=32 / default tile's speed,
  not fast, but no longer a hard failure. ~20 lines, no kernel-logic change.
- **A2 (proper, larger): make the kernel handle `group_size < BK`.** Generalize
  lines 374/529/602 to `groups_per_bk = ceildiv(BK, group_size)` scale-groups per
  BK-block (the reverse of the current "BK-blocks per group") and advance the
  scales iterator accordingly. This lets the BK=128 decode configs work with g32
  and is the prerequisite for a *fast* g32 path on these tiles. Touches the smem
  scales layout (`:806-807`, already `ceildiv(BK, group_size)`-based, so partly
  ready) and the three `group_size // BK` sites.

**Test.** Add a `group_size=32` case at N=K=4096 (and a `BK=128, group_size=32`
config) to `test/gpu/quantization/` so the compile path is covered; assert output
vs an fp32 dequant reference (L2 < 3e-2).

**Upstream framing.** `[Kernel][GPU] qmatmul: support group_size < block_k`
(A2) or `[Kernel][GPU] qmatmul: guard BK=128 configs against small group_size`
(A1). Reference the reproduction and the g128 assumption.

---

## B. Latent out-of-bounds A-tile load when BM > M — BUG (correctness, any GPU)

**Symptom (measured then retracted-as-headline).** With a config whose `BM > M`
at large K, the A-tile global load over-reads past the `[M, K]` A buffer. We hit
`CUDA_ERROR_ILLEGAL_ADDRESS` at K=14336 (down_proj) only when our *first* harness
**forced** BM=128; MAX's real dispatch uses BM=16 there and runs clean (43.6 µs).
So it does **not** manifest for the shipping Llama-3 dispatch — but it is a genuine
latent defect (the over-read exists at BM=16/M=1 too; it just lands in mapped
padding). Our "MAX crashes on down_proj" claim was retracted; this is the real,
narrower bug.

**Root cause (verified).** `qmatmul_gpu.mojo:823`
`var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx[1], bk_start)` tiles
A into `BM × BK` tiles with no runtime `M` bound; the async copy inside
`multistage_mma_q` (the `copy_dram_to_sram_async` A path, `:341-357`) then reads
all `BM` rows. When `block_idx[1]*BM + BM > M`, it reads `(BM - M_rem)*K` elements
past the buffer.

**Fix.** Mask the A global→shared async copy to rows `< M` (predicated load,
zero-fill the out-of-range rows), the standard boundary-tile handling — mirrors
how the dense GEMV clamps (`linalg/gemv.mojo:366` uses `min(row_base+r, last_row)`).
Concretely, thread the runtime `M` into the A copier and skip/zero rows `≥ M`. This
is correctness-only and independent of Q4_0 (fires for any quant matmul that
selects a `BM > M` config at large K).

**Test.** A quant-matmul case with `M = 1`, `BM = 128`, `K = 14336` under
`--config=asan` (or a guard page) so the over-read is caught; assert no OOB and
correct output.

**Upstream framing.** `[Kernel][GPU] multistage_mma_q: mask A-tile load for M < BM`.
**PR/issue-ready writeup with the exact fix (dense-path `_mask_tensor_row` pattern),
verified line numbers, and the test: [`reports/max-B-fix.md`](max-B-fix.md).**
Not built/tested here (no MAX bazel env); must be compiled + ASan-run upstream.

---

## C. No M=1-specialized decode-quant kernel — COVERAGE GAP (the real contribution)

**Symptom (measured).** For shapes not in the tuned cascade (gate_up N=28672,
lm_head N=128256), Q4_0 at M=1 falls to the generic `default_config` 128×128 GEMM
tile and runs at **~15 % of the memory roofline** (1/128 of the M-tile used); the
cascade shapes that *do* have a decode config (up_proj/down_proj, BM=16) reach
69–81 %. There is no single Q4_0 kernel that is uniformly near-roofline at M=1.
llama.cpp's `mul_mat_vec_q` is (73–100 %), and **our kernel matches it** (74–102 %,
parity — see `reports/h0-results.md`).

**Fix — DON'T build a competing kernel; [modular#6668](https://github.com/modular/modular/pull/6668)
already adds a fused dequant-GEMV** for batch-1 on pre-Blackwell NVIDIA (incl.
sm_86) — exactly this pattern, for **NVFP4**. So the M=1 decode-GEMV mechanism is
already coming upstream. Our only non-duplicating question: **does GGUF Q4_0 ride
that path, or only NVFP4?** Action = read #6668 to see if its dequant-GEMV is
format-general or E2M1-only, and *measure* whether Q4_0 at gate_up/lm_head still
falls to the 128×128 tile after #6668/#6708 land. If Q4_0 is left out, the niche is
the **Q4_0 wiring into #6668's GEMV**, not a new kernel. Our `kernels/q4_0_gemv.mojo`
(Q8_1 + `dp4a`, llama.cpp parity) then serves as a reference/benchmark for that
wiring, not as a competing upstream kernel.

**Caveat.** Our kernel is at *parity* with llama.cpp, not faster — and the decode-
GEMV pattern is already being upstreamed for NVFP4. So there is **no "write our own
kernel and upstream it" story left** unless Q4_0 turns out to be excluded from the
#6668 path; the honest contribution is coverage verification + Q4_0 wiring + tests.

**Test.** Extend `benchmarks/gpu/linalg` with the six Llama-3 Q4_0 decode shapes
at M=1 and gate a roofline-% regression check.

---

## D. Attention decode mid-context gap — NEEDS ONE MORE MEASUREMENT before any fix

**Symptom (measured).** MAX `mha_decoding` is at the roofline at long context
(86.7 % of spec at seq 16384) but only **61.7 %** at seq 4096, while FlashInfer
reaches **79.3 %** on the same GQA 32/8, hd128 shapes — an ~18-point gap.

**Why no fix is proposed yet.** The comparison is **confounded**: FlashInfer read
*contiguous* KV (`single_decode_with_kv_cache`), MAX reads *paged* KV (page 128).
The gap could be MAX's kernel OR the paging pattern.

**Prerequisite step (do this first).** Re-run FlashInfer with the **paged** path
(`BatchDecodeWithPagedKVCacheWrapper`, page_size 128) at the same shapes (the
`.venv-attn` + `bench/run_attention_flashinfer.py` harness is in place; add a
paged variant). Then:
- paged FlashInfer ≈ 79 % → real MAX `mha_decoding` mid-context inefficiency
  (investigate split-K partition count / occupancy at seq 4096 on 82 SMs) →
  propose an upstream fix.
- paged FlashInfer ≈ 62 % → the gap is the paged-read pattern; MAX's kernel is
  fine; no fix, just document.
Logged in `reports/open-questions.md`.

---

## Sequencing (after removing everything that already has a PR)

The "don't touch anything with a PR" rule leaves only three things that are ours:

1. **B — mask the A-tile load.** The single clean fix with no competing PR. Land
   standalone (separate modular clone → PR) with an ASAN / guard-page test.
2. **D-prerequisite — paged FlashInfer measurement.** Confirms or kills the
   attention gap before anyone proposes an attention change. Cheap; reuses
   `.venv-attn`.
3. **Track #6668 + #6708; when they land, verify Q4_0 on sm_86** — does g32 now
   compile, and does Q4_0 ride the fused dequant-GEMV or still hit the 128×128
   tile? Add the **Q4_0 tests** the PRs don't. Only if Q4_0 is excluded does any
   Q4_0-wiring work open up.

Everything else (the group<BK generalization, the decode-GEMV kernel itself) is
already in flight upstream as the #6668→#6708 chain — **we do not reimplement it.**
The H0 audit's job was to decide this from evidence, including checking open PRs
first, and it has: the net new work is one bug fix (B) and one measurement (D).
