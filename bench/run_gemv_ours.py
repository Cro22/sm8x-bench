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
    # Rebuild if EITHER the entry OR any imported kernel source is newer than the
    # binary. (Previously only ENTRY was checked, so kernel-only edits were
    # silently measured against a stale binary.)
    srcs = [ENTRY, *(_REPO / "kernels").glob("*.mojo")]
    newest_src = max(s.stat().st_mtime for s in srcs if s.exists())
    if BINARY.exists() and BINARY.stat().st_mtime >= newest_src:
        return
    print("building q4_0_gemv_ours ...")
    r = subprocess.run(["mojo", "build", "-I", "kernels", str(ENTRY),
                        "-o", str(BINARY)],
                       cwd=str(_REPO), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        raise SystemExit("mojo build failed")


def _wall_samples(out: str) -> list[float]:
    sm = _SAMPLES.search(out)
    if not sm:
        return []
    return [float(v) for v in sm[1].replace(" ", "").split(",") if v]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="o_proj")
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--passes", type=int, default=3,
                    help="independent nsys runs; the median run is reported and "
                         "ALL pass medians are preserved in timing.passes")
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

    l2_rel = max_abs = max_rel = None
    passes = []          # one entry per independent nsys run
    for pi in range(args.passes):
        print(f"profiling under nsys: ours {args.shape} Q4_0 M{M} "
              f"(pass {pi+1}/{args.passes}) ...")
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
        passes.append({
            "median_us": kt["med_us"], "min_us": kt["min_us"],
            "max_us": kt.get("max_us"), "nsys_mean_us": kt["avg_us"],
            "n_instances": sum(k["instances"] for k in kt["kernels"]),
            "kernels": kt["kernels"], "warning": kt["warning"],
            "observed_sm_clock": meta.get("__sm_clock_mhz__"),
            "wall_samples_us": [round(v, 3) for v in _wall_samples(out)],
        })

    pass_meds = sorted(pr["median_us"] for pr in passes)
    median_us = pass_meds[len(pass_meds) // 2]        # median run
    rep = min(passes, key=lambda pr: abs(pr["median_us"] - median_us))
    spread_pct = round(100 * (pass_meds[-1] - pass_meds[0]) / median_us, 2)

    bytes_moved = roofline.gemv_bytes(M, N, K, "Q4_0")
    flops = roofline.gemv_flops(M, N, K)
    kernel_names = ", ".join(k["name"] for k in rep["kernels"])
    note = (f"our Q4_0 GEMV; nsys kernel(s): {kernel_names}. bf16 activations. "
            f"median of {args.passes} passes {pass_meds} us "
            f"(spread {spread_pct}%); all preserved in timing.passes. ")
    if rep["warning"]:
        note += "WARN " + rep["warning"] + ". "

    path = write_result(
        impl="ours", kernel="gemv", variant=f"Q4_0_M{M}_{args.shape}",
        shape={"M": M, "N": N, "K": K},
        dtype={"weights": "Q4_0", "activations": "bf16", "accum": "fp32"},
        bytes_moved=bytes_moved, flops=flops,
        timing={"source": "nsys_gpukernsum",
                "n_instances": rep["n_instances"],
                "launches_per_sample": 0, "median_us": median_us,
                "min_us": rep["min_us"], "nsys_mean_us": rep["nsys_mean_us"],
                "passes": args.passes, "pass_medians_us": pass_meds,
                "pass_spread_pct": spread_pct,
                "pass_detail": passes,
                "harness_wallclock_median_us": (
                    round(st.median(rep["wall_samples_us"]), 3)
                    if rep["wall_samples_us"] else None),
                "kernels": rep["kernels"]},
        correctness={"validated": True, "l2_rel_err": l2_rel,
                     "max_abs_err": max_abs, "max_rel_err": max_rel,
                     "tolerance": "l2_rel<3e-2 (vs fp32 dequant ref)"},
        graphics_clock_mhz_locked=args.locked_clock, mem_clock_locked=False,
        observed_sm_clock=rep["observed_sm_clock"],
        measured_gbps=args.measured_gbps,
        inputs=reference.hash_inputs(p),
        notes=note, gpu_index=args.gpu_index,
    )
    print(f"wrote {path}")
    ag = roofline.achieved_gbps(bytes_moved, median_us)
    print(f"median-of-{args.passes} kernel={median_us:.2f} us  achieved={ag:.1f} GB/s  "
          f"pct_spec={100*ag/roofline.spec_bandwidth_gbps('sm_86'):.1f}%  "
          f"passes={pass_meds} spread={spread_pct}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
