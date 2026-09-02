"""Run the llama.cpp Q4_0 gemv_bench driver UNDER nsys, take the per-kernel
duration as authoritative timing, validate (the driver checks vs our fp32 ref),
and write the results JSON with impl="llamacpp". Mirrors bench/run_gemv_max.py.

The driver reads the SAME Q4_0 weight bytes MAX used (bench/inputs/), so the
comparison is on identical weights. llama.cpp uses f32 activations (ggml
mul_mat_vec_q); MAX used bf16 — noted in the JSON. Weight traffic (Q4_0,
0.5625 B/weight) is identical and dominates.

Usage:
    uv run python -m bench.run_gemv_llamacpp --shape o_proj --M 1 \
        --locked-clock 1695 --measured-gbps 815.5
"""

from __future__ import annotations

import argparse
import re
import statistics as st
import sys
from pathlib import Path

from bench import nsys, reference, roofline, shapes
from bench.results_io import write_result

_REPO = Path(__file__).resolve().parent.parent
DRIVER = _REPO / "bench" / "baselines" / "llamacpp" / "gemv_bench"
PIN = _REPO / "bench" / "baselines" / "llamacpp" / "PIN"

_CORR = re.compile(
    r"correctness:\s*(PASS|FAIL)\s+l2_rel_err=\s*([\d.eE+-]+)\s+"
    r"max_abs_err=\s*([\d.eE+-]+)\s+max_rel_err=\s*([\d.eE+-]+)")
_SAMPLES = re.compile(r"samples_us=\s*([\d.,eE+\- ]+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="o_proj")
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--fmt", default="Q4_0", choices=["Q4_0"])
    ap.add_argument("--locked-clock", type=int, default=None)
    ap.add_argument("--measured-gbps", type=float, default=None)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    if not DRIVER.exists():
        print(f"ABORT: driver not built at {DRIVER}", file=sys.stderr); return 1

    table = {n: (n, N, K) for (n, N, K) in shapes.GEMV_SHAPES}
    _, N, K = table[args.shape]
    M = args.M
    p = reference.gemv_paths(args.shape, args.fmt, M)
    if not (p["W"].exists() and p["x"].exists() and p["ref"].exists()):
        reference.gen_gemv(args.shape, N, K, args.fmt, M=M)

    pin = PIN.read_text().strip() if PIN.exists() else ""
    llamacpp_sha = pin.split("@")[-1].strip() if "@" in pin else pin

    # Disable ggml CUDA graphs so nsys sees every launch as a distinct kernel
    # instance (with graphs the whole replay collapses to 1 instance -> the
    # "median" would be a single observation, not comparable to ours/MAX's 120).
    cmd = ["env", "GGML_CUDA_DISABLE_GRAPHS=1", str(DRIVER), str(N), str(K), str(M),
           str(p["W"]), str(p["x"]), str(p["ref"])]
    print(f"profiling under nsys: llama.cpp {args.shape} Q4_0 M{M} ...")
    rows = nsys.kernel_summary(cmd, cwd=_REPO)
    meta = rows[0]
    out = meta.get("__stdout__", "")
    if meta.get("__returncode__", 1) != 0:
        print(out); print("ABORT: driver exited non-zero.", file=sys.stderr); return 1

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

    note = (f"llama.cpp Q4_0; nsys kernel(s): {kernel_names}. "
            f"f32 activations (MAX used bf16); identical Q4_0 weight bytes. ")
    if kt["warning"]:
        note += "WARN " + kt["warning"] + ". "
    if wall_median is not None:
        note += f"harness wall-clock median {wall_median} us. "

    path = write_result(
        impl="llamacpp",
        kernel="gemv",
        variant=f"Q4_0_M{M}_{args.shape}",
        shape={"M": M, "N": N, "K": K},
        dtype={"weights": "Q4_0", "activations": "fp32", "accum": "fp32"},
        bytes_moved=bytes_moved,
        flops=flops,
        timing={
            "source": "nsys_gpukernsum",
            "n_instances": sum(k["instances"] for k in kt["kernels"]),
            "launches_per_sample": 0,
            "median_us": median_us, "q1_us": 0, "q3_us": 0,
            "min_us": kt["min_us"], "p95_us": 0,
            "nsys_mean_us": kt["avg_us"],
            "harness_wallclock_median_us": wall_median,
            "kernels": kt["kernels"],
        },
        correctness={"validated": True, "l2_rel_err": l2_rel,
                     "max_abs_err": max_abs, "max_rel_err": max_rel,
                     "tolerance": "l2_rel<3e-2 (vs our fp32 dequant ref)"},
        graphics_clock_mhz_locked=args.locked_clock,
        mem_clock_locked=False,
        observed_sm_clock=meta.get("__sm_clock_mhz__"),
        measured_gbps=args.measured_gbps,
        llamacpp_sha=llamacpp_sha,
        notes=note,
        gpu_index=args.gpu_index,
    )
    print(f"wrote {path}")
    ag = roofline.achieved_gbps(bytes_moved, median_us)
    print(f"nsys kernel median={median_us:.2f} us  achieved={ag:.1f} GB/s  "
          f"pct_spec={100*ag/roofline.spec_bandwidth_gbps('sm_86'):.1f}%"
          + (f"  pct_measured={100*ag/args.measured_gbps:.1f}%" if args.measured_gbps else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
