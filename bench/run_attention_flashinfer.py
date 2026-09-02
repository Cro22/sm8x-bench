"""Run the FlashInfer decode-attention driver UNDER nsys (same per-kernel timing
as the MAX attention harness), validate, and write the results JSON with
impl="flashinfer". Single-request decode, GQA 32/8, head_dim 128, fp16 KV, batch
1, at the canonical seq lengths. Directly comparable at the roofline to MAX's
mha_decoding (both read the whole KV once).

The driver runs in .venv-attn (torch + flashinfer), a separate env from max.

Usage:
    uv run python -m bench.run_attention_flashinfer --seq 16384 \
        --locked-clock 1695 --measured-gbps 816.3
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

from bench import nsys, roofline, shapes
from bench.results_io import write_result

_REPO = Path(__file__).resolve().parent.parent
VENV_PY = _REPO / ".venv-attn" / "bin" / "python"
DRIVER = _REPO / "bench" / "baselines" / "flashinfer" / "flashinfer_decode.py"

_CORR = re.compile(
    r"correctness:\s*(PASS|FAIL)\s+l2_rel_err=\s*([\d.eE+-]+)\s+"
    r"max_abs_err=\s*([\d.eE+-]+)\s+max_rel_err=\s*([\d.eE+-]+)")
_SAMPLES = re.compile(r"samples_us=\s*([\d.,eE+\- ]+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=16384, choices=shapes.ATTN_SEQ_LENS)
    ap.add_argument("--locked-clock", type=int, default=None)
    ap.add_argument("--measured-gbps", type=float, default=None)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    if not VENV_PY.exists():
        print(f"ABORT: {VENV_PY} missing (create .venv-attn with torch+flashinfer)"); return 1

    a = shapes.ATTN
    q_heads, kv_heads, head_dim = a["q_heads"], a["kv_heads"], a["head_dim"]
    seq = args.seq

    cmd = [str(VENV_PY), str(DRIVER), str(seq)]
    print(f"profiling under nsys: FlashInfer decode seq{seq} ...")
    rows = nsys.kernel_summary(cmd, cwd=_REPO,
                               env={"FLASHINFER_DISABLE_VERSION_CHECK": "1"})
    meta = rows[0]
    out = meta.get("__stdout__", "")
    if meta.get("__returncode__", 1) != 0:
        print(out); print("ABORT: driver exited non-zero."); return 1

    cm = _CORR.search(out)
    if not cm:
        print(out); print("ABORT: no correctness line."); return 1
    if cm[1] != "PASS":
        print("ABORT: correctness FAILED."); return 1
    l2_rel, max_abs, max_rel = float(cm[2]), float(cm[3]), float(cm[4])

    kt = nsys.per_invocation_us(rows)
    if kt["med_us"] is None:
        print("ABORT: nsys found no GPU kernels."); return 1

    wall_median = None
    sm = _SAMPLES.search(out)
    if sm:
        s = [float(v) for v in sm[1].replace(" ", "").split(",") if v]
        if s:
            wall_median = round(st.median(s), 3)

    median_us = kt["med_us"]
    bytes_moved = roofline.attention_decode_bytes(seq, kv_heads, head_dim)
    kernel_names = ", ".join(k["name"] for k in kt["kernels"])

    note = (f"FlashInfer single_decode_with_kv_cache; nsys kernel(s): "
            f"{kernel_names}. Contiguous KV (MAX reads paged, page 128); both read "
            f"the whole KV once so the roofline comparison holds. torch 2.14+cu130. ")
    if kt["warning"]:
        note += "WARN " + kt["warning"] + ". "
    if wall_median is not None:
        note += f"harness wall-clock median {wall_median} us. "

    path = write_result(
        impl="flashinfer",
        kernel="attention_decode",
        variant=f"fp16_gqa{q_heads}x{kv_heads}_hd{head_dim}_seq{seq}",
        shape={"batch": 1, "q_heads": q_heads, "kv_heads": kv_heads,
               "head_dim": head_dim, "seq_len": seq, "page_size": shapes.PAGE_SIZE},
        dtype={"q": "fp16", "kv": "fp16", "accum": "fp32"},
        bytes_moved=bytes_moved,
        flops=0,
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
                     "tolerance": "l2_rel<3e-2 (vs fp32 softmax ref)"},
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
