"""Measure MAX's Q4_0 GEMV through its REAL public dispatcher (matmul_gpu_qint4
[group_size=32] with static K/N), per shape. This REPLACES the earlier
run_gemv_max.py Q4_0 numbers, which forced multistage_gemm_q at 128x128x32 for
all shapes and therefore did not reflect what MAX dispatches.

For each shape it stamps comptime N,K into bench/mojo/qgemv_max_dispatch.mojo,
builds a per-shape binary, and:
  - build fails (o_proj/qkv: tuned config BK=128, g32 incompatible) -> record a
    "compile_fail" status result (honest N/A).
  - builds -> run under nsys (authoritative per-kernel timing, proven clock),
    validate L2, write the result. down_proj does NOT crash under the real
    (BM=16) config; that earlier crash was an artifact of the forced BM=128.

Usage:
    uv run python -m bench.run_gemv_max_dispatch --shape up_proj --M 1 \
        --locked-clock 1695 --measured-gbps 815.5
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from bench import nsys, reference, roofline, shapes
from bench.results_io import write_result, write_status

ENTRY = Path(__file__).resolve().parent / "mojo" / "qgemv_max_dispatch.mojo"
_REPO = Path(__file__).resolve().parent.parent

_CORR = re.compile(
    r"correctness:\s*(PASS|FAIL)\s+l2_rel_err=\s*([\d.eE+-]+)\s+"
    r"max_abs_err=\s*([\d.eE+-]+)\s+max_rel_err=\s*([\d.eE+-]+)")


def _stamp(N: int, K: int) -> None:
    src = ENTRY.read_text()
    src = re.sub(r"comptime N = \d+", f"comptime N = {N}", src, count=1)
    src = re.sub(r"comptime K = \d+", f"comptime K = {K}", src, count=1)
    ENTRY.write_text(src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="up_proj")
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

    _stamp(N, K)
    binary = ENTRY.parent / f"qgemv_max_dispatch_{args.shape}"
    print(f"building MAX real-dispatch for {args.shape} (N={N} K={K}) ...")
    b = subprocess.run(
        ["mojo", "build", "-I", "modular/max/kernels/src", str(ENTRY),
         "-o", str(binary)],
        cwd=str(_REPO), capture_output=True, text=True)
    if b.returncode != 0:
        # Expected for o_proj/qkv: tuned m<=16/m<=32 config is BK=128, and
        # group_size(32)//BK(128)==0 -> comptime failure. Real MAX limitation.
        tail = (b.stderr or b.stdout).strip().splitlines()
        reason = "group_size//BK==0 (BK=128 g128-tuned config)" if any(
            "int_tuple" in l or "out-of-bounds" in l for l in tail) else "build failed"
        path = write_status(
            impl="max", kernel="gemv", variant=f"Q4_0_M{M}_{args.shape}",
            shape={"M": M, "N": N, "K": K}, status="compile_fail",
            notes=f"matmul_gpu_qint4[g32] real dispatch does not compile: {reason}",
            gpu_index=args.gpu_index)
        print(f"COMPILE-FAIL (real MAX limitation). wrote {path}")
        return 0

    # harness argv order is: M N K W x ref [rtol] [atol]
    cmd = [str(binary), str(M), str(N), str(K),
           str(p["W"]), str(p["x"]), str(p["ref"])]
    print(f"profiling under nsys: MAX real-dispatch {args.shape} Q4_0 M{M} ...")
    rows = nsys.kernel_summary(cmd, cwd=_REPO)
    meta = rows[0]
    out = meta.get("__stdout__", "")
    if meta.get("__returncode__", 1) != 0:
        # A runtime crash (e.g. illegal address) — record honestly.
        crash = "CUDA_ERROR_ILLEGAL_ADDRESS" if "ILLEGAL_ADDRESS" in (out + (meta.get("__stdout__") or "")) else "runtime error"
        path = write_status(
            impl="max", kernel="gemv", variant=f"Q4_0_M{M}_{args.shape}",
            shape={"M": M, "N": N, "K": K}, status="crash",
            notes=f"real dispatch runtime failure: {crash}", gpu_index=args.gpu_index)
        print(out); print(f"CRASH. wrote {path}", file=sys.stderr)
        return 0

    cm = _CORR.search(out)
    if not cm or cm[1] != "PASS":
        print(out); print("ABORT: correctness not PASS.", file=sys.stderr); return 1
    l2_rel, max_abs, max_rel = float(cm[2]), float(cm[3]), float(cm[4])

    kt = nsys.per_invocation_us(rows)
    if kt["med_us"] is None:
        print("ABORT: nsys found no GPU kernels.", file=sys.stderr); return 1
    median_us = kt["med_us"]
    bytes_moved = roofline.gemv_bytes(M, N, K, "Q4_0")
    flops = roofline.gemv_flops(M, N, K)
    kernel_names = ", ".join(k["name"] for k in kt["kernels"])

    path = write_result(
        impl="max", kernel="gemv", variant=f"Q4_0_M{M}_{args.shape}",
        shape={"M": M, "N": N, "K": K},
        dtype={"weights": "Q4_0", "activations": "bf16", "accum": "fp32"},
        bytes_moved=bytes_moved, flops=flops,
        timing={"source": "nsys_gpukernsum",
                "n_instances": sum(k["instances"] for k in kt["kernels"]),
                "launches_per_sample": 0, "median_us": median_us,
                "q1_us": 0, "q3_us": 0, "min_us": kt["min_us"], "p95_us": 0,
                "nsys_mean_us": kt["avg_us"],
                "kernels": kt["kernels"]},
        correctness={"validated": True, "l2_rel_err": l2_rel,
                     "max_abs_err": max_abs, "max_rel_err": max_rel,
                     "tolerance": "l2_rel<3e-2 (vs fp32 dequant ref)"},
        graphics_clock_mhz_locked=args.locked_clock, mem_clock_locked=False,
        observed_sm_clock=meta.get("__sm_clock_mhz__"),
        measured_gbps=args.measured_gbps,
        notes=f"MAX REAL dispatch (matmul_gpu_qint4[g32], static K/N); kernel(s): {kernel_names}. ",
        gpu_index=args.gpu_index)
    print(f"wrote {path}")
    ag = roofline.achieved_gbps(bytes_moved, median_us)
    print(f"nsys kernel median={median_us:.2f} us  achieved={ag:.1f} GB/s  "
          f"pct_spec={100*ag/roofline.spec_bandwidth_gbps('sm_86'):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
