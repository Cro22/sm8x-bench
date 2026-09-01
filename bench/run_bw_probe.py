"""Run bench/mojo/bw_probe.mojo, parse its output, and write the measured-roofline
results JSON via the shared writer. Clocks must already be locked (this only
records the value you pass; it does not lock them — locking is done on the
Windows host: `nvidia-smi -lgc <clk>,<clk>`).

Usage:
    uv run python -m bench.run_bw_probe --locked-clock 1695
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from bench.results_io import write_result

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "bench" / "mojo" / "bw_probe.mojo"

_LINE = re.compile(
    r"measured_bw_gbps_best=\s*([\d.]+)\s+median=\s*([\d.]+)\s+"
    r"N_bytes=\s*(\d+)\s+K=\s*(\d+)"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-clock", type=int, default=None,
                    help="graphics clock (MHz) the GPU is locked to for this run")
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    proc = subprocess.run(
        ["mojo", "run", str(PROBE)], capture_output=True, text=True, cwd=REPO
    )
    out = proc.stdout
    print(out, end="")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    if "correctness: PASS" not in out:
        print("ABORT: probe correctness did not PASS; not writing a result.",
              file=sys.stderr)
        return 1

    m = _LINE.search(out)
    if not m:
        print("ABORT: could not parse bandwidth line.", file=sys.stderr)
        return 1
    best, median, n_bytes, k = (
        float(m[1]), float(m[2]), int(m[3]), int(m[4])
    )

    # best GB/s == fastest launch == min time; median GB/s == median time.
    median_us = n_bytes / (median * 1e9) * 1e6
    min_us = n_bytes / (best * 1e9) * 1e6

    path = write_result(
        impl="probe",
        kernel="bw_probe",
        variant="read_stream_2GiB",
        shape={"bytes_per_launch": n_bytes},
        dtype={"element": "int32"},
        bytes_moved=n_bytes,
        flops=0,
        timing={
            "n_samples": 30,
            "launches_per_sample": k,
            "median_us": round(median_us, 3),
            "q1_us": 0, "q3_us": 0,
            "min_us": round(min_us, 3),
            "p95_us": 0,
            "nsys_mean_us": 0,
        },
        correctness={"validated": True, "max_abs_err": 0, "max_rel_err": 0,
                     "tolerance": "exact: sum == N"},
        graphics_clock_mhz_locked=args.locked_clock,
        mem_clock_locked=False,
        measured_gbps=best,  # the probe's best IS the measured roofline
        notes=(
            "Read-only streaming reduction over 2 GiB; best-of-30. This value is "
            "the measured memory roofline other kernels' pct_measured references. "
            "Bandwidth is mem-clock bound; graphics-clock lock at "
            f"{args.locked_clock} MHz slightly under-reports peak vs boost, but is "
            "the consistent denominator since all kernels run at the same lock."
        ),
        gpu_index=args.gpu_index,
    )
    print(f"wrote {path}")
    print(f"measured roofline: best={best:.1f} GB/s  median={median:.1f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
