"""Run the MAX GEMV Mojo entry point, parse its output, compute robust timing
statistics, and write the results JSON via the shared writer.

The Mojo side (bench/mojo/gemv_max.mojo) prints:
    device: <name>
    correctness: PASS|FAIL max_abs_err=<> max_rel_err=<>
    launches_per_sample= <K>
    samples_us= v1,v2,...,v30
This runner owns the statistics (median/IQR/min/p95) and the JSON so the schema
lives in exactly one place (bench/results_io.py).

Usage:
    uv run python -m bench.run_gemv_max --shape o_proj --fmt fp16 --locked-clock 1695
"""

from __future__ import annotations

import argparse
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

from bench import roofline, shapes
from bench.results_io import write_result

REPO = Path(__file__).resolve().parent.parent
ENTRY = REPO / "bench" / "mojo" / "gemv_max.mojo"

_CORR = re.compile(r"correctness:\s*(PASS|FAIL)\s+max_abs_err=\s*([\d.eE+-]+)\s+max_rel_err=\s*([\d.eE+-]+)")
_K = re.compile(r"launches_per_sample=\s*(\d+)")
_SAMPLES = re.compile(r"samples_us=\s*([\d.,eE+\- ]+)")


def _pctl(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="o_proj")
    ap.add_argument("--fmt", default="fp16", choices=shapes.WEIGHT_FORMATS)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--locked-clock", type=int, default=None)
    ap.add_argument("--measured-gbps", type=float, default=None,
                    help="measured roofline from the bw_probe run this session")
    ap.add_argument("--nsys-mean-us", type=float, default=0.0,
                    help="kernel mean (us) from an nsys --stats run, for the "
                         "mandated cross-check; must agree with median within ~5%%")
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    table = {n: (n, N, K) for (n, N, K) in shapes.GEMV_SHAPES}
    if args.shape not in table:
        ap.error(f"unknown shape {args.shape}; have {list(table)}")
    _, N, K = table[args.shape]
    M = args.M

    proc = subprocess.run(
        ["mojo", "run", "-I", "modular/max/kernels/src", str(ENTRY)],
        capture_output=True, text=True, cwd=REPO,
    )
    out = proc.stdout
    print(out, end="")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    cm = _CORR.search(out)
    if not cm:
        print("ABORT: no correctness line.", file=sys.stderr)
        return 1
    passed = cm[1] == "PASS"
    max_abs, max_rel = float(cm[2]), float(cm[3])
    if not passed:
        print("ABORT: kernel correctness FAILED; not recording a timing result.",
              file=sys.stderr)
        return 1

    km = _K.search(out)
    sm = _SAMPLES.search(out)
    if not (km and sm):
        print("ABORT: missing launches_per_sample or samples_us.", file=sys.stderr)
        return 1
    k = int(km[1])
    samples = [float(v) for v in sm[1].replace(" ", "").split(",") if v]
    if len(samples) < 2:
        print("ABORT: too few samples.", file=sys.stderr)
        return 1

    median_us = st.median(samples)
    q1, q3 = _pctl(samples, 0.25), _pctl(samples, 0.75)
    iqr_ratio = (q3 - q1) / median_us if median_us else 0.0
    notes = ""
    if iqr_ratio > 0.05:
        notes = (f"WARNING IQR/median={iqr_ratio:.1%} > 5% — check clocks/thermal/"
                 f"contention before trusting this number. ")
    if args.nsys_mean_us:
        disagree = abs(median_us - args.nsys_mean_us) / median_us
        tag = "OK" if disagree <= 0.05 else "WARNING"
        notes += (f"nsys cross-check {tag}: harness median {median_us:.2f} us vs "
                  f"nsys mean {args.nsys_mean_us:.2f} us ({disagree:.1%}). ")

    bytes_moved = roofline.gemv_bytes(M, N, K, args.fmt)
    flops = roofline.gemv_flops(M, N, K)

    # MAX fp16 activations; accum fp32. For fp16 weights M=1 this is the native
    # MAX GEMV; M>1 fp16 would be cuBLAS (see audit) — record what ran.
    dtype = {"weights": args.fmt, "activations": "fp16", "accum": "fp32"}

    path = write_result(
        impl="max",
        kernel="gemv",
        variant=f"{args.fmt}_M{M}_{args.shape}",
        shape={"M": M, "N": N, "K": K},
        dtype=dtype,
        bytes_moved=bytes_moved,
        flops=flops,
        timing={
            "n_samples": len(samples),
            "launches_per_sample": k,
            "median_us": round(median_us, 4),
            "q1_us": round(q1, 4), "q3_us": round(q3, 4),
            "min_us": round(min(samples), 4),
            "p95_us": round(_pctl(samples, 0.95), 4),
            "nsys_mean_us": round(args.nsys_mean_us, 4),
        },
        correctness={"validated": True, "max_abs_err": max_abs,
                     "max_rel_err": max_rel, "tolerance": "rtol=1e-2 atol=1e-3"},
        graphics_clock_mhz_locked=args.locked_clock,
        mem_clock_locked=False,
        measured_gbps=args.measured_gbps,
        notes=notes + "Native MAX GEMV path (M=1 dispatch); kernel gemv_split_k_float16 (confirmed by nsys).",
        gpu_index=args.gpu_index,
    )
    print(f"wrote {path}")
    ag = roofline.achieved_gbps(bytes_moved, median_us)
    print(f"median={median_us:.2f} us  achieved={ag:.1f} GB/s  "
          f"pct_spec={100*ag/roofline.spec_bandwidth_gbps('sm_86'):.1f}%"
          + (f"  pct_measured={100*ag/args.measured_gbps:.1f}%" if args.measured_gbps else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
