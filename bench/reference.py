"""Deterministic inputs + fp32 reference for every (kernel, shape, format).

Generated ONCE here and written to `bench/inputs/` as raw little-endian .bin
(no header) plus a .json sidecar with shape/dtype. Every implementation — the
Mojo MAX harness and the CUDA baselines — reads these exact bytes, so
cross-implementation validation compares identical inputs against one reference.

The reference is always fp32: weights dequantized exactly per format, matmul in
fp32 with fp32 activations cast from the fp16 inputs (see mojo-gpu-kernel skill).

Usage:
    uv run python -m bench.reference gemv --shape o_proj --fmt fp16
    uv run python -m bench.reference gemv --all --fmt fp16
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np

from bench import shapes

INPUTS_DIR = Path(__file__).resolve().parent / "inputs"

# Activation dtype each MAX weight-format path actually consumes (from the audit):
# fp16 weights -> fp16 acts; bf16 weights and the int4 (Q4_0) kernel -> bf16 acts.
ACT_DTYPE = {
    "fp16": np.float16,
    "bf16": ml_dtypes.bfloat16,
    "Q8_0": ml_dtypes.bfloat16,
    "Q4_0": ml_dtypes.bfloat16,
    "Q4_K": ml_dtypes.bfloat16,
}
_ACT_TAG = {np.float16: "fp16", ml_dtypes.bfloat16: "bf16"}


def gemv_paths(name: str, fmt: str, M: int) -> dict:
    """Canonical input/reference file paths for a GEMV config. The single source
    of truth for filenames — reference.py writes them, run_gemv_max.py reads them.
    W is M-independent (not duplicated per M); x depends on activation dtype+M;
    ref depends on fmt+M."""
    act_tag = _ACT_TAG[ACT_DTYPE[fmt]]
    return {
        "act_tag": act_tag,
        "W": INPUTS_DIR / f"gemv_{name}_{fmt}_W.bin",
        "x": INPUTS_DIR / f"gemv_{name}_{act_tag}_M{M}_x.bin",
        "ref": INPUTS_DIR / f"gemv_{name}_{fmt}_M{M}_ref.bin",
    }


def _rng(*tags) -> np.random.Generator:
    # Stable per-(shape,fmt,...) stream derived from the canonical SEED. Uses a
    # process-STABLE hash (hashlib) — Python's builtin hash() is randomized per
    # process (PYTHONHASHSEED), which would make the same tensor differ between
    # invocations and silently break W/ref consistency across separate runs.
    key = f"{shapes.SEED}|" + "|".join(str(t) for t in tags)
    digest = hashlib.sha256(key.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _save(path: Path, arr: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def gen_gemv(name: str, N: int, K: int, fmt: str, M: int = 1) -> None:
    """Write W (N x K, `fmt`), x (M x K, activation dtype for `fmt`), and the
    fp32 reference y (M x N). The reference is computed from the EXACT bytes the
    kernel sees (weights rounded/dequantized per format, activations in their
    real dtype), so correctness reflects only kernel error, not input drift.

    y = W @ x^T semantics for weights stored N x K (transpose_b path):
    ref[m, n] = sum_k dequant(W)[n, k] * xf32[m, k].
    """
    p = gemv_paths(name, fmt, M)
    act_dt = ACT_DTYPE[fmt]
    act_tag = p["act_tag"]

    # Activations: dtype depends on the weight format's kernel. Shared across all
    # formats with the same activation dtype for this (shape, M) (stable seed).
    xr = _rng("x", name, M, K, act_tag)
    x_act = xr.standard_normal((M, K)).astype(act_dt)
    x_f32 = x_act.astype(np.float32)
    _save(p["x"], x_act, {"tensor": "x", "shape": [M, K], "dtype": act_tag})

    wr = _rng("w", name, N, K)
    if fmt in ("fp16", "bf16"):
        w_dt = np.float16 if fmt == "fp16" else ml_dtypes.bfloat16
        W = wr.standard_normal((N, K)).astype(w_dt)
        W_f32 = W.astype(np.float32)  # exact dequant of the stored low-precision W
        _save(p["W"], W,
              {"tensor": "W", "shape": [N, K], "dtype": fmt, "format": fmt})
    else:
        raise NotImplementedError(
            f"format {fmt!r} not implemented yet (Q4_0 next; needs the exact "
            f"GGUF byte layout — see gguf-quant-formats skill)")

    ref = (x_f32 @ W_f32.T).astype(np.float32)  # (M, N)
    _save(p["ref"], ref,
          {"tensor": "ref_y", "shape": [M, N], "dtype": "float32",
           "format": fmt, "activations": act_tag,
           "note": "fp32 y = x_f32 @ dequant(W)_f32^T"})

    print(f"wrote {p['W'].name}  {p['x'].name}  {p['ref'].name}  "
          f"(N={N} K={K} M={M} fmt={fmt} acts={act_tag})")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kernel", required=True)
    g = sub.add_parser("gemv")
    g.add_argument("--shape", default=None,
                   help="shape name from shapes.GEMV_SHAPES (e.g. o_proj)")
    g.add_argument("--all", action="store_true", help="all GEMV shapes")
    g.add_argument("--fmt", default="fp16", choices=shapes.WEIGHT_FORMATS)
    g.add_argument("--M", type=int, default=1)
    args = ap.parse_args()

    if args.kernel == "gemv":
        table = {n: (n, N, K) for (n, N, K) in shapes.GEMV_SHAPES}
        if args.all:
            targets = list(table.values())
        elif args.shape:
            if args.shape not in table:
                ap.error(f"unknown shape {args.shape}; have {list(table)}")
            targets = [table[args.shape]]
        else:
            ap.error("pass --shape <name> or --all")
        for name, N, K in targets:
            gen_gemv(name, N, K, args.fmt, M=args.M)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
