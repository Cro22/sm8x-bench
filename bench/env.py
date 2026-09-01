"""Capture the environment for a results JSON: GPU state (pynvml) and toolchain
provenance (versions + git SHAs). Fills the `gpu` and `env` blocks of the
results schema in the bench-methodology skill. Never hand-edit a results file;
always let this run.

Usage:
    uv run python -m bench.env            # pretty-print captured env
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULAR_DIR = REPO_ROOT / "modular"

# Compute-capability -> short arch string we use everywhere.
_ARCH = {(8, 6): "sm_86", (8, 9): "sm_89", (8, 0): "sm_80", (9, 0): "sm_90"}


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=30
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def git_sha(repo: Path) -> str:
    sha = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    if sha is None:
        return "unknown"
    dirty = _run(["git", "status", "--porcelain"], cwd=repo)
    return sha + ("-dirty" if dirty else "")


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def mojo_version() -> str:
    # `mojo --version` -> "Mojo 1.0.0 (ed45d567)". Prefer the package metadata
    # (fast, no subprocess) and fall back to the binary.
    v = _pkg_version("mojo")
    if v != "unknown":
        return v
    out = _run(["mojo", "--version"])
    return out or "unknown"


def cuda_driver_version(pynvml) -> str:
    """CUDA version the driver supports (e.g. '12.6'), from NVML."""
    try:
        raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
    except Exception:
        try:
            raw = pynvml.nvmlSystemGetCudaDriverVersion()
        except Exception:
            return "unknown"
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def capture_gpu(index: int = 0) -> dict:
    """GPU block for the results JSON. Reads live clocks; the caller must have
    already locked clocks (scripts/gpu-lock.sh) for a valid measurement."""
    import pynvml

    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
        arch = _ARCH.get((major, minor), f"sm_{major}{minor}")

        sm_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
        max_sm = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        max_mem = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)

        # Is the graphics/SM clock locked? NVML exposes the applied gpu-lock
        # range on most GeForce driver builds; treat "current == a pinned value"
        # loosely and record the raw numbers so the report can decide.
        try:
            locked = bool(pynvml.nvmlDeviceGetGpcClkVfOffset(h) is not None)
        except Exception:
            locked = None

        try:
            power_w = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        except Exception:
            power_w = None

        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()

        # SM count: NVML has no portable direct query; use the audited constant
        # table keyed by arch and cross-check the name.
        from bench.roofline import HARDWARE

        sms = HARDWARE.get(arch, {}).get("sms")

        return {
            "name": name,
            "arch": arch,
            "compute_capability": f"{major}.{minor}",
            "sms": sms,
            "driver": driver,
            "cuda_driver_version": cuda_driver_version(pynvml),
            "graphics_clock_mhz_current": sm_clock,
            "graphics_clock_mhz_max": max_sm,
            # Filled by the caller/harness from gpu-lock.sh; current == the value
            # you locked to if the lock succeeded.
            "graphics_clock_mhz_locked": None,
            "mem_clock_mhz": mem_clock,
            "mem_clock_mhz_max": max_mem,
            "mem_clock_locked": False,  # GeForce rejects -lmc; default false
            "power_limit_w": power_w,
        }
    finally:
        pynvml.nvmlShutdown()


def capture_env(
    llamacpp_sha: str = "",
    flashinfer_version: str = "",
) -> dict:
    """env block for the results JSON."""
    import pynvml

    pynvml.nvmlInit()
    try:
        cuda = cuda_driver_version(pynvml)
    finally:
        pynvml.nvmlShutdown()

    return {
        "mojo_version": mojo_version(),
        "max_version": _pkg_version("max"),
        "modular_sha": git_sha(MODULAR_DIR),
        "cuda_version": cuda,
        "llamacpp_sha": llamacpp_sha,
        "flashinfer_version": flashinfer_version or _pkg_version("flashinfer"),
        "harness_sha": git_sha(REPO_ROOT),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def capture(index: int = 0) -> dict:
    return {"gpu": capture_gpu(index), "env": capture_env()}


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))
