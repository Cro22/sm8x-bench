"""Single writer for every results JSON. The schema is defined in the
bench-methodology skill; this module is the only place that emits it, so no
harness ever hand-rolls (or drifts) the format. Never hand-edit a results file.

The `gpu` and `env` blocks are filled automatically from bench.env. The caller
passes the measured fields; roofline percentages are computed from
bench.roofline so bytes/percentages are never hand-typed.

Usage (from a runner):
    from bench.results_io import write_result
    path = write_result(
        impl="max", kernel="gemv", variant="Q4_0",
        shape={"M":1,"N":4096,"K":4096},
        dtype={"weights":"Q4_0","activations":"bf16","accum":"fp32"},
        bytes_moved=..., flops=...,
        timing={...}, correctness={...},
        graphics_clock_mhz_locked=1695,
        measured_gbps=795.8,   # from the bw_probe run in the same session
        notes="...")
"""

from __future__ import annotations

import json
from pathlib import Path

from bench import env as _env
from bench import roofline as _roofline

SCHEMA_VERSION = 1
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_VALID_IMPL = {"max", "llamacpp", "flashinfer", "cublas", "ours", "probe"}
_VALID_KERNEL = {"gemv", "attention_decode", "matmul", "bw_probe"}


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(s)).strip("-")


def write_result(
    *,
    impl: str,
    kernel: str,
    variant: str,
    shape: dict,
    dtype: dict,
    timing: dict,
    correctness: dict,
    bytes_moved: int,
    flops: int = 0,
    graphics_clock_mhz_locked: int | None = None,
    mem_clock_locked: bool = False,
    observed_sm_clock: dict | None = None,
    measured_gbps: float | None = None,
    llamacpp_sha: str = "",
    flashinfer_version: str = "",
    notes: str = "",
    gpu_index: int = 0,
) -> Path:
    """Assemble, validate, and write one results JSON. Returns the path."""
    if impl not in _VALID_IMPL:
        raise ValueError(f"impl {impl!r} not in {_VALID_IMPL}")
    if kernel not in _VALID_KERNEL:
        raise ValueError(f"kernel {kernel!r} not in {_VALID_KERNEL}")

    gpu = _env.capture_gpu(gpu_index)
    envblock = _env.capture_env(
        llamacpp_sha=llamacpp_sha, flashinfer_version=flashinfer_version
    )
    if graphics_clock_mhz_locked is not None:
        gpu["graphics_clock_mhz_locked"] = graphics_clock_mhz_locked
    gpu["mem_clock_locked"] = mem_clock_locked
    # Provenance: the SM clock actually observed under load during the timed run
    # (from bench.nsys sampling). A locked GPU holds this; deviation = the run
    # was NOT at the claimed clock. This replaces trusting the CLI arg.
    if observed_sm_clock is not None:
        gpu["graphics_clock_mhz_observed"] = observed_sm_clock
        med = observed_sm_clock.get("median")
        lk = graphics_clock_mhz_locked
        if med is not None and lk and abs(med - lk) / lk > 0.03:
            gpu["clock_lock_warning"] = (
                f"observed SM clock median {med} MHz != locked {lk} MHz "
                f"(range {observed_sm_clock.get('min')}-{observed_sm_clock.get('max')})")

    arch = gpu["arch"]
    median_us = timing["median_us"]

    achieved_gbps = _roofline.achieved_gbps(bytes_moved, median_us)
    achieved_tflops = _roofline.achieved_tflops(flops, median_us) if flops else 0.0
    roof = _roofline.roofline_block(bytes_moved, median_us, arch, measured_gbps)

    record = {
        "schema": SCHEMA_VERSION,
        "impl": impl,
        "kernel": kernel,
        "variant": variant,
        "gpu": gpu,
        "shape": shape,
        "dtype": dtype,
        "bytes_moved": bytes_moved,
        "flops": flops,
        "timing": timing,
        "achieved_gbps": round(achieved_gbps, 2),
        "achieved_tflops": round(achieved_tflops, 3),
        "roofline": roof,
        "correctness": correctness,
        "env": envblock,
        "notes": notes,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = envblock["timestamp_utc"].replace(":", "").replace("-", "").split(".")[0]
    fname = f"{_slug(impl)}_{_slug(kernel)}_{_slug(variant)}_{_slug(gpu['arch'])}_{ts}.json"
    path = RESULTS_DIR / fname
    path.write_text(json.dumps(record, indent=2))
    return path
