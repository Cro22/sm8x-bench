#!/usr/bin/env bash
# Lock/unlock GPU clocks for benchmarking. Usage: gpu-lock.sh {lock [sm_clock]|unlock|status}
# Defaults: 3090 -> 1695, 4090 -> 2520. Lower the clock if the run hits the power limit.
set -euo pipefail
cmd="${1:-status}"
name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$name" in
  *3090*) default_clk=1695 ;;
  *4090*) default_clk=2520 ;;
  *)      default_clk="" ;;
esac
case "$cmd" in
  lock)
    clk="${2:-$default_clk}"
    [ -n "$clk" ] || { echo "unknown GPU '$name', pass a clock explicitly"; exit 1; }
    sudo nvidia-smi -pm 1
    sudo nvidia-smi -lgc "$clk,$clk"
    if sudo nvidia-smi -lmc "$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1)" 2>/dev/null; then
      echo "mem clock locked"
    else
      echo "mem clock lock not supported on this GPU (expected on GeForce) — record mem_clock_locked=false"
    fi
    ;;
  unlock)
    sudo nvidia-smi -rgc || true
    sudo nvidia-smi -rmc || true
    ;;
  status)
    nvidia-smi --query-gpu=name,driver_version,persistence_mode,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.limit,temperature.gpu --format=csv
    nvidia-smi -q -d PERFORMANCE | sed -n '/Clocks Event Reasons/,/^$/p' || true
    ;;
  *) echo "usage: $0 {lock [sm_clock]|unlock|status}"; exit 1 ;;
esac
