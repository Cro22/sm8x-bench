"""Run OUR Q4_0 GEMV kernel UNDER nsys (authoritative per-kernel timing),
validate (the entry checks vs our fp32 dequant ref), write impl="ours" JSON.
Same inputs and same measurement as the MAX and llama.cpp Q4_0 runners, so the
three are directly comparable. Mirrors bench/run_gemv_max.py.

Usage:
    uv run python -m bench.run_gemv_ours --shape o_proj --M 1 \
        --locked-clock 1695 --measured-gbps 815.5
"""

from __future__ import annotations

import argparse
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

from bench import nsys, reference, roofline, shapes
from bench.results_io import write_result

ENTRY = Path(__file__).resolve().parent / "mojo" / "q4_0_gemv_ours.mojo"
BINARY = Path(__file__).resolve().parent / "mojo" / "q4_0_gemv_ours"
_REPO = Path(__file__).resolve().parent.parent

_CORR = re.compile(
    r"correctness:\s*(PASS|FAIL)\s+l2_rel_err=\s*([\d.eE+-]+)\s+"
    r"max_abs_err=\s*([\d.eE+-]+)\s+max_rel_err=\s*([\d.eE+-]+)")
_SAMPLES = re.compile(r"samples_us=\s*([\d.,eE+\- ]+)")


def ensure_built() -> None:
    if BINARY.exists() and BINARY.stat().st_mtime >= ENTRY.stat().st_mtime:
        return
    print("building q4_0_gemv_ours ...")
    r = subprocess.run(["mojo", "build", "-I", "kernels", str(ENTRY),
                        "-o", str(BINARY)],
                       cwd=str(_REPO), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        raise SystemExit("mojo build failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="o_proj")
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--locked-clock", type=int, default=None)
    ap.add_argument("--measured-gbps", type=float, default=None)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    table = {n: (n, N, K) for (n, N, K) in shapes.GEMV_SHAPES}
    _, N, K = table[args.shape]
    M = args.M
    p = reference.gemv_paths(args.shape, "Q4_0", M)
    if not (p["W"].exists() and p["x"].exists() and p["ref"].exists()):
        reference.gen_gemv(args.shape, N, K, "Q4_0", M=M)

    ensure_built()
    cmd = [str(BINARY), str(N), str(K), str(M),
           str(p["W"]), str(p["x"]), str(p["ref"]), "3e-2", "5e-3"]
    print(f"profiling under nsys: ours {args.shape} Q4_0 M{M} ...")
    rows = nsys.kernel_summary(cmd, cwd=_REPO)
    meta = rows[0]
    out = meta.get("__stdout__", "")
    if meta.get("__returncode__", 1) != 0:
        print(out); print("ABORT: entry exited non-zero.", file=sys.stderr); return 1
    cm = _CORR.search(out)
    if not cm:
        print(out); print("ABORT: no correctness line.", file=sys.stderr); return 1
    if cm[1] != "PASS":
        print("ABORT: correctness FAILED.", file=sys.stderr); return 1
    l2_rel, max_abs, max_rel = float(cm[2]), float(cm[3]), float(cm[4])

    kt = nsys.per_invocation_us(rows)
    if kt["med_us"] is None:
        print("ABORT: nsys found no GPU kernels.", file=sys.stderr); return 1

    wall_median = None
    sm = _SAMPLES.search(out)
    if sm:
        s = [float(v) for v in sm[1].replace(" ", "").split(",") if v]
        if s:
            wall_median = round(st.median(s), 3)

    median_us = kt["med_us"]
    bytes_moved = roofline.gemv_bytes(M, N, K, "Q4_0")
    flops = roofline.gemv_flops(M, N, K)
    kernel_names = ", ".join(k["name"] for k in kt["kernels"])
    note = f"our Q4_0 GEMV; nsys kernel(s): {kernel_names}. bf16 activations. "
    if kt["warning"]:
        note += "WARN " + kt["warning"] + ". "
    if wall_median is not None:
        note += f"harness wall-clock median {wall_median} us. "

    path = write_result(
        impl="ours", kernel="gemv", variant=f"Q4_0_M{M}_{args.shape}",
        shape={"M": M, "N": N, "K": K},
        dtype={"weights": "Q4_0", "activations": "bf16", "accum": "fp32"},
        bytes_moved=bytes_moved, flops=flops,
        timing={"source": "nsys_gpukernsum",
                "n_instances": sum(k["instances"] for k in kt["kernels"]),
                "launches_per_sample": 0, "median_us": median_us,
                "q1_us": 0, "q3_us": 0, "min_us": kt["min_us"], "p95_us": 0,
                "nsys_mean_us": kt["avg_us"],
                "harness_wallclock_median_us": wall_median,
                "kernels": kt["kernels"]},
        correctness={"validated": True, "l2_rel_err": l2_rel,
                     "max_abs_err": max_abs, "max_rel_err": max_rel,
                     "tolerance": "l2_rel<3e-2 (vs fp32 dequant ref)"},
        graphics_clock_mhz_locked=args.locked_clock, mem_clock_locked=False,
        measured_gbps=args.measured_gbps, notes=note, gpu_index=args.gpu_index,
    )
    print(f"wrote {path}")
    ag = roofline.achieved_gbps(bytes_moved, median_us)
    print(f"nsys kernel median={median_us:.2f} us  achieved={ag:.1f} GB/s  "
          f"pct_spec={100*ag/roofline.spec_bandwidth_gbps('sm_86'):.1f}%"
          + (f"  pct_measured={100*ag/args.measured_gbps:.1f}%" if args.measured_gbps else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
