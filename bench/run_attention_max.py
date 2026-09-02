"""Run the MAX flash-decoding attention entry point UNDER nsys, take the
per-kernel duration as the authoritative timing, validate correctness (the entry
compares the decode kernel against MAX's mha_gpu_naive), and write the results
JSON. Mirrors bench/run_gemv_max.py.

Decode config (bench/shapes.py): batch 1, 32 q-heads, 8 kv-heads, head_dim 128,
paged KV page_size 128, KV fp16. seq_len sweeps {1024, 4096, 16384}.

Usage:
    uv run python -m bench.run_attention_max --seq 4096 --locked-clock 1695 \
        --measured-gbps 815.5
"""

from __future__ import annotations

import argparse
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

from bench import nsys, roofline, shapes
from bench.results_io import write_result

ENTRY = Path(__file__).resolve().parent / "mojo" / "attn_max.mojo"
BINARY = Path(__file__).resolve().parent / "mojo" / "attn_max"
_REPO = Path(__file__).resolve().parent.parent

_CORR = re.compile(
    r"correctness:\s*(PASS|FAIL)\s+l2_rel_err=\s*([\d.eE+-]+)\s+"
    r"max_abs_err=\s*([\d.eE+-]+)\s+max_rel_err=\s*([\d.eE+-]+)")
_SAMPLES = re.compile(r"samples_us=\s*([\d.,eE+\- ]+)")


def ensure_built() -> None:
    if BINARY.exists() and BINARY.stat().st_mtime >= ENTRY.stat().st_mtime:
        return
    print("building attn_max ...")
    r = subprocess.run(
        ["mojo", "build", "-I", "modular/max/kernels/src",
         str(ENTRY), "-o", str(BINARY)],
        cwd=str(_REPO), capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        raise SystemExit("mojo build failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True,
                    help="KV context length (one of shapes.ATTN_SEQ_LENS)")
    ap.add_argument("--kv-dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--locked-clock", type=int, default=None)
    ap.add_argument("--measured-gbps", type=float, default=None)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    ensure_built()
    cmd = [str(BINARY), str(args.seq), "2e-2", "2e-3"]
    print(f"profiling under nsys: attention decode seq={args.seq} ...")
    rows = nsys.kernel_summary(cmd, cwd=_REPO)
    meta = rows[0]
    out = meta.get("__stdout__", "")
    if meta.get("__returncode__", 1) != 0:
        print(out); print("ABORT: entry exited non-zero.", file=sys.stderr); return 1

    cm = _CORR.search(out)
    if not cm:
        print(out); print("ABORT: no correctness line.", file=sys.stderr); return 1
    if cm[1] != "PASS":
        print("ABORT: correctness FAILED; not recording.", file=sys.stderr); return 1
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
    q_heads, kv_heads, head_dim = shapes.Q_HEADS, shapes.KV_HEADS, shapes.HEAD_DIM
    bytes_moved = roofline.attention_decode_bytes(args.seq, kv_heads, head_dim, 2)
    # flops: QK^T + softmax*V, per query token, batch 1 -> ~4 * seq * q_heads * head_dim
    flops = 4 * args.seq * q_heads * head_dim
    kernel_names = ", ".join(k["name"] for k in kt["kernels"])

    note = f"nsys-authoritative; kernel(s): {kernel_names}. correctness vs mha_gpu_naive. "
    if kt["warning"]:
        note += "WARN " + kt["warning"] + ". "
    if wall_median is not None:
        note += f"harness wall-clock median {wall_median} us ({wall_median/median_us:.1f}x nsys). "

    path = write_result(
        impl="max",
        kernel="attention_decode",
        variant=f"{args.kv_dtype}_gqa32x8_hd128_seq{args.seq}",
        shape={"batch": 1, "q_heads": q_heads, "kv_heads": kv_heads,
               "head_dim": head_dim, "seq_len": args.seq, "page_size": shapes.PAGE_SIZE},
        dtype={"kv": args.kv_dtype, "q": args.kv_dtype, "accum": "fp32"},
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
                     "tolerance": "l2_rel<2e-2 (vs mha_gpu_naive)"},
        graphics_clock_mhz_locked=args.locked_clock,
        mem_clock_locked=False,
        observed_sm_clock=meta.get("__sm_clock_mhz__"),
        measured_gbps=args.measured_gbps,
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
