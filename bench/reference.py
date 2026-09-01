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
import json
from pathlib import Path

import numpy as np

from bench import shapes

INPUTS_DIR = Path(__file__).resolve().parent / "inputs"


def _rng(*tags) -> np.random.Generator:
    # Stable per-(shape,fmt) stream derived from the canonical SEED so each
    # tensor is reproducible and independent.
    mix = shapes.SEED
    for t in tags:
        mix = (mix * 1000003 + hash(str(t))) & 0xFFFFFFFFFFFF
    return np.random.default_rng(mix)


def _save(path: Path, arr: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def gen_gemv(name: str, N: int, K: int, fmt: str, M: int = 1) -> None:
    """Write W (N x K, `fmt`), x (M x K, fp16), and the fp32 reference y (M x N).

    y = W @ x^T semantics for weights stored N x K (transpose_b path):
    ref[m, n] = sum_k Wf32[n, k] * xf32[m, k].
    """
    stem = INPUTS_DIR / f"gemv_{name}_{fmt}"

    # Activations: fp16, shared across all formats for this shape.
    xr = _rng("x", name, M, K)
    x_f16 = xr.standard_normal((M, K)).astype(np.float16)
    _save(stem.with_name(f"gemv_{name}_x").with_suffix(".bin"), x_f16,
          {"tensor": "x", "shape": [M, K], "dtype": "float16"})

    wr = _rng("w", name, N, K)
    if fmt == "fp16":
        W_f16 = wr.standard_normal((N, K)).astype(np.float16)
        W_f32 = W_f16.astype(np.float32)
        _save(Path(f"{stem}_W.bin"), W_f16,
              {"tensor": "W", "shape": [N, K], "dtype": "float16",
               "format": "fp16"})
    else:
        raise NotImplementedError(
            f"format {fmt!r} not implemented yet (quant formats: Q4_0 next; "
            f"see gguf-quant-formats skill for exact byte layout)")

    # fp32 reference: cast activations to fp32, matmul in fp32.
    ref = (x_f16.astype(np.float32) @ W_f32.T).astype(np.float32)  # (M, N)
    _save(Path(f"{stem}_ref.bin"), ref,
          {"tensor": "ref_y", "shape": [M, N], "dtype": "float32",
           "format": fmt, "note": "fp32 y = x_f32 @ W_f32^T"})

    print(f"wrote {stem}_W.bin  gemv_{name}_x.bin  {stem}_ref.bin  "
          f"(N={N} K={K} M={M} fmt={fmt})")


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
