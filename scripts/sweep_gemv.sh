#!/usr/bin/env bash
# GEMV sweep: all Llama-3-8B GEMV shapes x {fp16,bf16} x M in {1,8}, MAX kernels,
# nsys-authoritative timing. Run from repo root. Clocks must be locked on the
# Windows host first (nvidia-smi -lgc 1695,1695); nsys kernel duration is robust
# to desktop contention but a quiet GPU is still preferable.
#
# Usage: scripts/sweep_gemv.sh [measured_gbps] [locked_clock]
set -uo pipefail
cd "$(dirname "$0")/.."

MEASURED="${1:-815.5}"
CLK="${2:-1695}"
SHAPES=(o_proj qkv_fused down_proj up_proj gate_up lm_head)
FMTS=(fp16 bf16)
MS=(1 8)

echo "=== GEMV sweep: measured_roofline=${MEASURED} GB/s, locked_clock=${CLK} MHz ==="
ok=0; fail=0
for shape in "${SHAPES[@]}"; do
  for fmt in "${FMTS[@]}"; do
    for m in "${MS[@]}"; do
      echo "--- $shape $fmt M$m ---"
      if uv run python -m bench.run_gemv_max --shape "$shape" --fmt "$fmt" --M "$m" \
           --locked-clock "$CLK" --measured-gbps "$MEASURED" \
           2>&1 | grep -E "nsys kernel median=|ABORT|correctness FAIL|wrote "; then
        ok=$((ok+1))
      else
        fail=$((fail+1)); echo "  (config failed)"
      fi
    done
  done
done
echo "=== sweep done: $ok ok, $fail failed ==="
