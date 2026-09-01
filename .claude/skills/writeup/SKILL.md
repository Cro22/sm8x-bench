---
name: writeup
description: How to turn results JSON files into the reports in reports/ (h0-results.md, README tables, later kernel writeups) — table generation from JSON only, roofline framing, honest treatment of anomalies, and the structure a Modular kernel engineer expects. Use whenever asked to write, update, or summarize results, produce a README section, draft a forum post from findings, or communicate benchmark outcomes in prose.
---

# Writing up results

The audience is a Modular kernel engineer skimming on a phone and a local-LLM
user deciding whether to care. Both want the same thing: the number, the
roofline, the conditions, and no spin.

## Tables come from JSON, never from memory

`bench/report.py` reads `bench/results/*.json` and emits Markdown tables. Reports
are regenerated, not edited by hand, except for the prose sections. If a number
is not in a JSON file it is not in the report. Each table row links to its JSON
file (relative path).

Standard table, one per (GPU, kernel):

| Impl | Variant | Shape | Median µs | IQR µs | GB/s | % spec | % measured | Validated |
|---|---|---|---|---|---|---|---|---|

Always include the `bw_probe` row so the "measured roofline" denominator is
visible. Sort by shape then by impl. Flag L2-resident shapes on the 4090 with a
footnote (see `bench-methodology`).

## Report structure (`reports/h0-results.md`)

1. **One-paragraph result.** What MAX achieves on consumer Ampere/Ada relative to
   roofline and to llama.cpp/FlashInfer, in two sentences, with the single most
   important number.
2. **Setup.** GPUs, driver, locked clocks, Mojo/MAX SHA, llama.cpp SHA,
   FlashInfer version, date. Link to `scripts/gpu-lock.sh` and the methodology
   skill so anyone can reproduce.
3. **Tables** per GPU per kernel.
4. **Reading the results.** Where MAX is at parity, where it is behind, where
   it is ahead. Tie each observation to a table row. Propose a mechanism only
   when there is evidence (nsys/ncu, config table, code) and label it as a
   hypothesis otherwise.
5. **Anomalies and caveats.** Everything that did not go cleanly: throttling,
   things that would not build, N/A cells, unexpected kernel dispatch. This
   section is what makes the rest believable.
6. **What this implies for H1.** Short. Points at `reports/audit.md` gaps.
7. **Reproduce.** Exact commands.

## Prose rules

- Percent of roofline is the headline metric; speedups vs another impl are
  secondary and always accompanied by roofline.
- No adjectives about performance ("blazing", "impressive"). State the number.
- Say "at parity" when within the IQR overlap; do not call a 3% difference a
  win for either side.
- If Mojo/MAX loses, say so plainly and say what the evidence suggests about
  why. That is more useful to Modular than a diplomatic table.
- Distinguish "MAX kernel X" from "MAX as shipped launches kernel X for this
  shape" — the second is what users experience.
- English, short sentences, no emoji.

## README

The README carries: the one-paragraph result, the two headline tables (GEMV
Q4 and attention decode, best GPU), the roofline chart if any, and links to the
full reports. Nothing else about the project's ambitions. The project is
described by what it measured.

## Forum posts and upstream issues

Draft in `reports/drafts/`, never post. Format: one paragraph of context, the
table row(s) in question, the file:line in upstream you believe is relevant,
the specific question. Modular staff respond to specific, reproducible
questions with numbers attached and ignore vague ones.
