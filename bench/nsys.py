"""Run a command under nsys and parse the CUDA GPU Kernel Summary.

nsys per-kernel duration is the AUTHORITATIVE timing for this repo (decided
2026-09-01): it measures the kernel(s) MAX (or a CUDA baseline) actually
launches, excluding host-side dispatch overhead (which is amortized in
production because the graph compiles once) and desktop-contention gaps. Every
implementation — MAX, cuBLAS, llama.cpp, FlashInfer — appears as GPU kernels in
nsys, so this is the one uniform, apples-to-apples measurement.

Parses the `gpukernsum` table rows:
  Time(%)  TotalTime(ns)  Instances  Avg(ns)  Med(ns)  Min(ns)  Max(ns)  StdDev  Name
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

# A gpukernsum data row: 8 numeric columns then the kernel name (may contain
# spaces/specials, so grab the rest of the line).
_ROW = re.compile(
    r"^\s*([\d.]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,.]+)\s+([\d,.]+)\s+"
    r"([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+(\S.*?)\s*$"
)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def kernel_summary(cmd: list[str], cwd: Path | str | None = None,
                   timeout: int = 900) -> list[dict]:
    """Run `cmd` under nsys, then generate the GPU kernel summary explicitly and
    parse it. Returns the GPU kernel rows, each:
    {name, instances, total_ns, avg_ns, med_ns, min_ns, max_ns, stddev_ns}.
    The first element is {"__stdout__": <program stdout>, "__returncode__": ...}
    so the caller can read correctness lines etc.

    Two nsys calls (profile, then `stats --report`) rather than one
    `--stats=true`: the explicit stats report is deterministic to parse and does
    not depend on how --stats output interleaves with the program's stdout."""
    with tempfile.TemporaryDirectory() as td:
        rep = Path(td) / "prof"
        # CUDA-only trace, no CPU sampling / context-switch tracing. On WSL the
        # default full trace makes nsys hang/crawl for minutes in teardown; with
        # these flags a profile completes in ~2 s. We only need GPU kernel
        # durations anyway.
        proc = subprocess.run(
            ["nsys", "profile", "--force-overwrite=true",
             "--trace=cuda", "--sample=none", "--cpuctxsw=none",
             "-o", str(rep), *cmd],
            capture_output=True, text=True, cwd=str(cwd) if cwd else None,
            timeout=timeout,
        )
        repfile = rep.with_suffix(".nsys-rep")
        stats = subprocess.run(
            ["nsys", "stats", "--force-export=true",
             "--report", "cuda_gpu_kern_sum", str(repfile)],
            capture_output=True, text=True, timeout=timeout,
        )
        text = stats.stdout + "\n" + stats.stderr

    rows: list[dict] = [{"__stdout__": proc.stdout, "__returncode__": proc.returncode}]
    # The kernel summary section is titled with "gpukernsum" or
    # "CUDA GPU Kernel Summary"; rows follow the header line. We match any data
    # row shaped like the kernel table and whose name isn't a memory op.
    in_kernels = False
    for line in text.splitlines():
        if "gpukernsum" in line or "CUDA GPU Kernel Summary" in line:
            in_kernels = True
            continue
        # A new section header (e.g. gpumemtimesum) ends the kernel table.
        if in_kernels and ("gpumem" in line or "Memory Operation" in line):
            in_kernels = False
        if not in_kernels:
            continue
        m = _ROW.match(line)
        if not m:
            continue
        name = m[9].strip()
        if name.startswith("[CUDA") or name in ("Name",):
            continue
        rows.append({
            "name": name,
            "total_ns": _num(m[2]),
            "instances": int(_num(m[3])),
            "avg_ns": _num(m[4]),
            "med_ns": _num(m[5]),
            "min_ns": _num(m[6]),
            "max_ns": _num(m[7]),
            "stddev_ns": _num(m[8]),
        })
    return rows


def per_invocation_us(rows: list[dict]) -> dict:
    """Collapse kernel rows into the per-matmul/op kernel time in microseconds.

    Assumes each distinct kernel fires once per op (true for gemv[/+reduce],
    gemm, qgemm, mha_decoding[/+splitk_reduce]). per-op time = sum over kernels
    of their per-instance time. Returns avg/med/min us + the kernel list and a
    warning if instance counts differ (hinting a kernel fires !=1x per op)."""
    kernels = [r for r in rows if "name" in r]
    if not kernels:
        return {"avg_us": None, "med_us": None, "min_us": None,
                "kernels": [], "warning": "no GPU kernels found in nsys output"}
    counts = {k["instances"] for k in kernels}
    warn = ""
    if len(counts) > 1:
        warn = (f"kernel instance counts differ {sorted(counts)} — a kernel may "
                f"fire !=1x per op; per-op sum may be approximate")
    avg = sum(k["avg_ns"] for k in kernels) / 1000.0
    med = sum(k["med_ns"] for k in kernels) / 1000.0
    mn = sum(k["min_ns"] for k in kernels) / 1000.0
    return {
        "avg_us": round(avg, 4), "med_us": round(med, 4), "min_us": round(mn, 4),
        "kernels": [{"name": k["name"], "instances": k["instances"],
                     "avg_us": round(k["avg_ns"] / 1000.0, 4)} for k in kernels],
        "warning": warn,
    }
