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


def _sample_sm_clock_until(proc, timeout: int) -> dict:
    """Poll clocks.sm while `proc` runs; return {min,max,median,n} MHz observed.
    This proves the graphics clock during the actual measurement (a locked GPU
    holds its value under load; an idle/boosting reading exposes an unlocked run)."""
    import time as _t
    samples: list[int] = []
    start = _t.monotonic()
    while proc.poll() is None and (_t.monotonic() - start) < timeout:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            v = r.stdout.strip().splitlines()
            if v:
                samples.append(int(float(v[0])))
        except Exception:
            pass
    if not samples:
        return {"min": None, "max": None, "median": None, "n": 0}
    samples.sort()
    return {"min": samples[0], "max": samples[-1],
            "median": samples[len(samples) // 2], "n": len(samples)}


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
        # durations anyway. We Popen (not run) so we can sample the actual SM
        # clock WHILE the kernel executes — provenance that the clock really was
        # locked, rather than trusting the CLI arg or an idle post-hoc reading.
        proc = subprocess.Popen(
            ["nsys", "profile", "--force-overwrite=true",
             "--trace=cuda", "--sample=none", "--cpuctxsw=none",
             "-o", str(rep), *cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(cwd) if cwd else None,
        )
        clk_samples = _sample_sm_clock_until(proc, timeout)
        out, err = proc.communicate()
        repfile = rep.with_suffix(".nsys-rep")
        stats = subprocess.run(
            ["nsys", "stats", "--force-export=true",
             "--report", "cuda_gpu_kern_sum", str(repfile)],
            capture_output=True, text=True, timeout=timeout,
        )
        text = stats.stdout + "\n" + stats.stderr

    rows: list[dict] = [{"__stdout__": out, "__returncode__": proc.returncode,
                         "__sm_clock_mhz__": clk_samples}]
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
    """Collapse kernel rows into the per-op kernel time in microseconds.

    The profiled run launches the timed op N times but also, once, any SETUP
    kernels (e.g. the Q4_0 weight repack). Per-op kernels all share the same high
    instance count (= number of launches); one-time setup kernels have a much
    smaller count. We keep only the per-op kernels (instances >= half the max
    count) and sum their per-instance time; setup kernels are excluded.
    (gemv[/+reduce], gemm, qgemm[/+splitk_reduce] all fire once per op.)"""
    kernels = [r for r in rows if "name" in r]
    if not kernels:
        return {"avg_us": None, "med_us": None, "min_us": None,
                "kernels": [], "excluded": [], "warning": "no GPU kernels found"}
    max_count = max(k["instances"] for k in kernels)
    per_op = [k for k in kernels if k["instances"] >= max_count / 2]
    setup = [k for k in kernels if k["instances"] < max_count / 2]
    counts = {k["instances"] for k in per_op}
    warn = ""
    if len(counts) > 1:
        warn = (f"per-op kernel instance counts differ {sorted(counts)} — a "
                f"kernel may fire !=1x per op; per-op sum may be approximate")
    avg = sum(k["avg_ns"] for k in per_op) / 1000.0
    med = sum(k["med_ns"] for k in per_op) / 1000.0
    mn = sum(k["min_ns"] for k in per_op) / 1000.0
    return {
        "avg_us": round(avg, 4), "med_us": round(med, 4), "min_us": round(mn, 4),
        "kernels": [{"name": k["name"], "instances": k["instances"],
                     "avg_us": round(k["avg_ns"] / 1000.0, 4)} for k in per_op],
        "excluded": [{"name": k["name"], "instances": k["instances"],
                      "avg_us": round(k["avg_ns"] / 1000.0, 4)} for k in setup],
        "warning": warn,
    }
