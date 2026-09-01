"""Run the MAX GEMV/GEMM entry point UNDER nsys, take the per-kernel duration as
the authoritative timing, validate correctness, and write the results JSON.

Timing policy (decided 2026-09-01): nsys per-kernel duration is the headline
number (bench/nsys.py). The Mojo harness' own wall-clock median is kept as a
secondary "dispatch-inclusive" reference — for M=1 it agrees with nsys; for M>1
it is dominated by host-side dispatch overhead that production amortizes.

Usage:
    uv run python -m bench.run_gemv_max --shape o_proj --fmt fp16 --M 1 \
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

ENTRY = Path(__file__).resolve().parent / "mojo" / "gemv_max.mojo"
BINARY = Path(__file__).resolve().parent / "mojo" / "gemv_max"
_REPO = Path(__file__).resolve().parent.parent


def ensure_built() -> None:
    """Build the entry to a binary once (rebuild if the source is newer), so
    nsys runs don't pay ~60 s of Mojo recompile per config."""
    if BINARY.exists() and BINARY.stat().st_mtime >= ENTRY.stat().st_mtime:
        return
    print("building gemv_max ...")
    r = subprocess.run(
        ["mojo", "build", "-I", "modular/max/kernels/src",
         str(ENTRY), "-o", str(BINARY)],
        cwd=str(_REPO), capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        raise SystemExit("mojo build failed")

_CORR = re.compile(r"correctness:\s*(PASS|FAIL)\s+max_abs_err=\s*([\d.eE+-]+)\s+max_rel_err=\s*([\d.eE+-]+)")
_SAMPLES = re.compile(r"samples_us=\s*([\d.,eE+\- ]+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="o_proj")
    ap.add_argument("--fmt", default="fp16", choices=shapes.WEIGHT_FORMATS)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--locked-clock", type=int, default=None)
    ap.add_argument("--measured-gbps", type=float, default=None)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    table = {n: (n, N, K) for (n, N, K) in shapes.GEMV_SHAPES}
    if args.shape not in table:
        ap.error(f"unknown shape {args.shape}; have {list(table)}")
    _, N, K = table[args.shape]
    M = args.M

    p = reference.gemv_paths(args.shape, args.fmt, M)
    if not (p["W"].exists() and p["x"].exists() and p["ref"].exists()):
        print(f"generating inputs for {args.shape} {args.fmt} M{M} ...")
        reference.gen_gemv(args.shape, N, K, args.fmt, M=M)

    ensure_built()
    cmd = [str(BINARY), str(M), str(N), str(K), args.fmt,
           str(p["W"]), str(p["x"]), str(p["ref"])]
    print(f"profiling under nsys: {args.shape} {args.fmt} M{M} ...")
    rows = nsys.kernel_summary(cmd, cwd=_REPO)
    meta = rows[0]
    out = meta.get("__stdout__", "")
    if meta.get("__returncode__", 1) != 0:
        print(out)
        print("ABORT: profiled program exited non-zero.", file=sys.stderr)
        return 1

    cm = _CORR.search(out)
    if not cm:
        print(out)
        print("ABORT: no correctness line.", file=sys.stderr)
        return 1
    if cm[1] != "PASS":
        print("ABORT: correctness FAILED; not recording a timing result.",
              file=sys.stderr)
        return 1
    max_abs, max_rel = float(cm[2]), float(cm[3])

    kt = nsys.per_invocation_us(rows)
    if kt["med_us"] is None:
        print("ABORT: nsys found no GPU kernels.", file=sys.stderr)
        return 1

    # Secondary: harness wall-clock median (dispatch-inclusive).
    wall_median = None
    sm = _SAMPLES.search(out)
    if sm:
        samples = [float(v) for v in sm[1].replace(" ", "").split(",") if v]
        if samples:
            wall_median = round(st.median(samples), 3)

    median_us = kt["med_us"]  # AUTHORITATIVE = nsys per-kernel median
    bytes_moved = roofline.gemv_bytes(M, N, K, args.fmt)
    flops = roofline.gemv_flops(M, N, K)
    kernel_names = ", ".join(k["name"] for k in kt["kernels"])

    note = f"nsys-authoritative timing; kernel(s): {kernel_names}. "
    if kt["warning"]:
        note += "WARN " + kt["warning"] + ". "
    if wall_median is not None:
        ratio = wall_median / median_us if median_us else 0
        note += (f"harness wall-clock median {wall_median} us "
                 f"({ratio:.1f}x nsys — dispatch overhead if >>1). ")

    path = write_result(
        impl="max",
        kernel="gemv",
        variant=f"{args.fmt}_M{M}_{args.shape}",
        shape={"M": M, "N": N, "K": K},
        dtype={"weights": args.fmt,
               "activations": "fp16" if args.fmt == "fp16" else "bf16",
               "accum": "fp32"},
        bytes_moved=bytes_moved,
        flops=flops,
        timing={
            "source": "nsys_gpukernsum",
            "n_instances": sum(k["instances"] for k in kt["kernels"]),
            "launches_per_sample": 0,
            "median_us": median_us,
            "q1_us": 0, "q3_us": 0,
            "min_us": kt["min_us"],
            "p95_us": 0,
            "nsys_mean_us": kt["avg_us"],
            "harness_wallclock_median_us": wall_median,
            "kernels": kt["kernels"],
        },
        correctness={"validated": True, "max_abs_err": max_abs,
                     "max_rel_err": max_rel, "tolerance": "rtol=1e-2 atol=1e-3"},
        graphics_clock_mhz_locked=args.locked_clock,
        mem_clock_locked=False,
        measured_gbps=args.measured_gbps,
        notes=note,
        gpu_index=args.gpu_index,
    )
    print(f"wrote {path}")
    ag = roofline.achieved_gbps(bytes_moved, median_us)
    at = roofline.achieved_tflops(flops, median_us)
    print(f"nsys kernel median={median_us:.2f} us  "
          f"achieved={ag:.1f} GB/s ({at:.1f} TFLOP/s)  "
          f"pct_spec={100*ag/roofline.spec_bandwidth_gbps('sm_86'):.1f}%"
          + (f"  pct_measured={100*ag/args.measured_gbps:.1f}%" if args.measured_gbps else ""))
    if wall_median:
        print(f"  (harness wall-clock median {wall_median:.1f} us = "
              f"{wall_median/median_us:.1f}x nsys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
