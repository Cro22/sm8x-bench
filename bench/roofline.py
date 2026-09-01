"""Spec + measured roofline, and minimum-traffic byte counts per kernel.

All bytes/flops for reports come from here — never hand-computed in prose.
Shapes come from bench.shapes; hardware constants are below (verified against
references/hardware.md and cross-checked with `nvidia-smi -q` on the card).

Usage:
    uv run python -m bench.roofline        # dump spec roofline + shape byte tables
"""

from __future__ import annotations

from bench import shapes

# --- Hardware constants (references/hardware.md) --------------------------
# spec_bw_gbps = mem_data_rate_gbps * bus_bits / 8
# fp32 FLOPS at a given clock = sms * 128 FMA/clk * 2 * clock_hz
HARDWARE = {
    "sm_86": {  # RTX 3090, GA102
        "name": "RTX 3090",
        "sms": 82,
        "boost_mhz": 1695,
        "mem_data_rate_gbps": 19.5,
        "bus_bits": 384,
        "spec_bw_gbps": 936.0,
        "l2_mb": 6,
        "power_w": 350,
        "fp32_tflops_boost": 35.6,
    },
    "sm_89": {  # RTX 4090, AD102
        "name": "RTX 4090",
        "sms": 128,
        "boost_mhz": 2520,
        "mem_data_rate_gbps": 21.0,
        "bus_bits": 384,
        "spec_bw_gbps": 1008.0,
        "l2_mb": 72,
        "power_w": 450,
        "fp32_tflops_boost": 82.6,
    },
}

# --- Weight-format byte cost ----------------------------------------------
# Minimum traffic per weight, in bytes. Quant formats use the *exact* GGUF
# block byte size (see gguf-quant-formats skill), not a rounded bits/weight.
#   Q8_0: 34 B / 32 weights   Q4_0: 18 B / 32   Q4_K: 144 B / 256
_BLOCK = {
    "fp16": (2.0, 1),
    "bf16": (2.0, 1),
    "Q8_0": (34.0, 32),
    "Q4_0": (18.0, 32),
    "Q4_K": (144.0, 256),
}


def bytes_per_weight(fmt: str) -> float:
    block_bytes, block_elems = _BLOCK[fmt]
    return block_bytes / block_elems


# --- Minimum traffic per kernel -------------------------------------------
def gemv_bytes(M: int, N: int, K: int, weight_fmt: str, act_bytes: int = 2) -> int:
    """Minimum DRAM traffic for y = W @ x, W stored N x K in `weight_fmt`.

    weights: N*K * bytes_per_weight(fmt)
    x: M*K activations (fp16 -> 2 B)   y: M*N outputs (fp16 -> 2 B)
    At M=1 the weight term dominates; x/y kept for completeness and larger M.
    """
    w = N * K * bytes_per_weight(weight_fmt)
    x = M * K * act_bytes
    y = M * N * act_bytes
    return int(round(w + x + y))


def gemv_flops(M: int, N: int, K: int) -> int:
    """2*M*N*K multiply-adds for the GEMV/GEMM."""
    return 2 * M * N * K


def attention_decode_bytes(
    seq_len: int, kv_heads: int, head_dim: int, kv_bytes: int = 2
) -> int:
    """Minimum DRAM traffic for one decode step (batch 1, 1 query token):
    read the whole K and V caches once.

    K and V each: seq_len * kv_heads * head_dim elements -> factor 2 for K+V.
    Q (kv_heads*q_per_kv*head_dim) and O are negligible vs the cache and omitted.

    NOTE: the bench-methodology skill writes this as
    `seq_len * kv_heads * head_dim * 2 (K) * 2 (K and V) * bytes`, which reads as
    two factors of 2 (=4x). That double-counts; the minimum traffic is a single
    read of K plus a single read of V = 2x. We use 2x (K+V) here as the
    algorithmic minimum and flagged the wording in reports/open-questions.md.
    """
    return 2 * seq_len * kv_heads * head_dim * kv_bytes


# --- Spec roofline ---------------------------------------------------------
def spec_bandwidth_gbps(arch: str) -> float:
    return HARDWARE[arch]["spec_bw_gbps"]


def spec_fp32_tflops(arch: str, clock_mhz: float | None = None) -> float:
    """FP32 roofline at the *locked* clock. Defaults to boost if clock unknown."""
    hw = HARDWARE[arch]
    clk_hz = (clock_mhz or hw["boost_mhz"]) * 1e6
    return hw["sms"] * 128 * 2 * clk_hz / 1e12


# --- Achieved-side helpers -------------------------------------------------
def achieved_gbps(bytes_moved: int, median_us: float) -> float:
    return bytes_moved / (median_us * 1e-6) / 1e9


def achieved_tflops(flops: int, median_us: float) -> float:
    return flops / (median_us * 1e-6) / 1e12


def roofline_block(
    bytes_moved: int,
    median_us: float,
    arch: str,
    measured_gbps: float | None = None,
) -> dict:
    """Fill the `roofline` block of the results JSON. `measured_gbps` comes from
    bw_probe run in the same session; None until the probe exists."""
    spec = spec_bandwidth_gbps(arch)
    ach = achieved_gbps(bytes_moved, median_us)
    return {
        "spec_gbps": spec,
        "measured_gbps": measured_gbps,
        "pct_spec": round(100 * ach / spec, 1),
        "pct_measured": (
            round(100 * ach / measured_gbps, 1) if measured_gbps else None
        ),
    }


if __name__ == "__main__":
    for arch, hw in HARDWARE.items():
        print(f"\n=== {arch}  {hw['name']} ===")
        print(f"  spec bandwidth : {spec_bandwidth_gbps(arch):.0f} GB/s")
        print(
            f"  fp32 @ boost   : {spec_fp32_tflops(arch):.1f} TFLOPS "
            f"({hw['boost_mhz']} MHz, {hw['sms']} SMs)"
        )
    print("\n=== GEMV min traffic (M=1), Llama-3-8B shapes ===")
    print(f"{'shape':<12}{'N':>7}{'K':>7}  " + "".join(f"{f:>12}" for f in shapes.WEIGHT_FORMATS))
    for name, N, K in shapes.GEMV_SHAPES:
        row = f"{name:<12}{N:>7}{K:>7}  "
        for fmt in shapes.WEIGHT_FORMATS:
            mb = gemv_bytes(1, N, K, fmt) / 1e6
            row += f"{mb:>10.2f}MB"
        print(row)
    print("\n=== decode attention KV traffic (GQA 8 kv-heads, head_dim 128, fp16) ===")
    for s in shapes.ATTN_SEQ_LENS:
        mb = attention_decode_bytes(s, shapes.KV_HEADS, shapes.HEAD_DIM) / 1e6
        print(f"  seq_len {s:>6}: {mb:>8.2f} MB")
