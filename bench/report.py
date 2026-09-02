"""Generate Markdown result tables from bench/results/*.json.

Tables are GENERATED, never hand-typed (writeup skill): if a number is not in a
JSON it is not in the report. Each row links to its JSON. Emits one table per
(gpu arch, kernel). Prose lives in reports/h0-results.md; this fills the tables.

Usage:
    uv run python -m bench.report               # print all tables to stdout
    uv run python -m bench.report > reports/_tables.md
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
REPO = Path(__file__).resolve().parent.parent


def _shape_str(kernel: str, sh: dict) -> str:
    if kernel == "gemv":
        return f"M{sh.get('M')} {sh.get('N')}x{sh.get('K')}"
    if kernel == "attention_decode":
        return f"seq{sh.get('seq_len')} gqa{sh.get('q_heads')}x{sh.get('kv_heads')} hd{sh.get('head_dim')}"
    if kernel == "bw_probe":
        return f"{sh.get('bytes_per_launch', 0)//(1024*1024)}MiB read"
    return json.dumps(sh)


def _sort_key(kernel: str, sh: dict) -> tuple:
    if kernel == "gemv":
        return (sh.get("M", 0), sh.get("N", 0), sh.get("K", 0))
    if kernel == "attention_decode":
        return (sh.get("seq_len", 0),)
    return (0,)


def load() -> list[dict]:
    out = []
    for f in sorted(RESULTS.glob("*.json")):
        d = json.load(open(f))
        d["_file"] = f.relative_to(REPO).as_posix()
        out.append(d)
    return out


def table(records: list[dict], arch: str, kernel: str) -> str:
    rows = [r for r in records if r["gpu"]["arch"] == arch and r["kernel"] == kernel]
    if not rows:
        return ""
    rows.sort(key=lambda r: (_sort_key(kernel, r["shape"]), r["impl"], r.get("variant", "")))
    hdr = ("| Impl | Variant | Shape | Median µs | min µs | GB/s | % spec "
           "| % meas | L2 err | Validated | JSON |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for r in rows:
        t, rf, c = r["timing"], r["roofline"], r["correctness"]
        # Non-measured outcome (compile-fail, crash) -> honest N/A row with reason.
        if r.get("status") or t.get("median_us") is None:
            reason = r.get("status", "n/a").replace("_", "-")
            lines.append(
                f"| {r['impl']} | {r.get('variant','')} | {_shape_str(kernel, r['shape'])} "
                f"| {reason} | — | — | — | — | — | — | [json]({r['_file']}) |")
            continue
        l2 = c.get("l2_rel_err")
        med = t["median_us"]
        ag = r.get("achieved_gbps")
        ps = rf.get("pct_spec")
        lines.append(
            f"| {r['impl']} | {r.get('variant','')} | {_shape_str(kernel, r['shape'])} "
            f"| {med:.2f} | {t.get('min_us',0):.2f} "
            f"| {('%.0f' % ag) if ag is not None else '—'} "
            f"| {('%.1f' % ps) if ps is not None else '—'} "
            f"| {('%.1f' % rf['pct_measured']) if rf.get('pct_measured') is not None else '—'} "
            f"| {('%.1e' % l2) if l2 is not None else '—'} "
            f"| {'yes' if c.get('validated') else 'NO'} "
            f"| [json]({r['_file']}) |"
        )
    return "\n".join(lines)


def main() -> int:
    records = load()
    archs = sorted({r["gpu"]["arch"] for r in records})
    kernels = ["bw_probe", "gemv", "attention_decode", "matmul"]
    for arch in archs:
        name = next((r["gpu"]["name"] for r in records if r["gpu"]["arch"] == arch), arch)
        print(f"\n### {name} ({arch})\n")
        for k in kernels:
            tbl = table(records, arch, k)
            if tbl:
                title = {"bw_probe": "Measured roofline (bandwidth probe)",
                         "gemv": "GEMV / matmul",
                         "attention_decode": "Attention decode",
                         "matmul": "Matmul"}.get(k, k)
                print(f"\n#### {title}\n")
                print(tbl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
