# Hardware constants (verify against `nvidia-smi -q` on the actual card)

| | RTX 3090 | RTX 4090 |
|---|---|---|
| Chip / arch | GA102 / Ampere / sm_86 | AD102 / Ada / sm_89 |
| SMs | 82 | 128 |
| CUDA cores | 10496 | 16384 |
| Boost clock (spec) | 1695 MHz | 2520 MHz |
| Base clock | 1395 MHz | 2235 MHz |
| Memory | 24 GB GDDR6X, 384-bit | 24 GB GDDR6X, 384-bit |
| Memory data rate | 19.5 Gbps | 21 Gbps |
| Spec bandwidth | 936 GB/s | 1008 GB/s |
| L2 | 6 MB | 72 MB |
| Shared mem per SM (max configurable) | 100 KB (128 KB L1+smem) | 100 KB (128 KB L1+smem) |
| FP32 (at boost) | 35.6 TFLOPS | 82.6 TFLOPS |
| FP16 tensor, dense (at boost) | ~71 TFLOPS | ~165 TFLOPS |
| FP8 | no | yes |
| Tensor core ISA | mma.sync (m16n8k16) | mma.sync (m16n8k16) |
| cp.async | yes (sm_80+) | yes |
| TMA / wgmma | no | no |
| Default power limit | 350 W | 450 W |

Notes:

- FP32 FLOPS = SMs × 128 FMA/clk × 2 × clock. Recompute at the *locked* clock;
  locking to 1695 on the 3090 gives 35.6 TFLOPS, locking lower scales down.
- Spec bandwidth = data_rate × bus_width / 8. Memory clock cannot usually be
  locked on GeForce; it sits at max under load, so spec is the right denominator
  but the measured probe is what you actually get (expect ~850–900 GB/s on the
  3090, ~900–950 on the 4090).
- The 4090's 72 MB L2 makes repeated-launch microbenchmarks of anything under
  ~50 MB partially L2-resident. Always include shapes larger than L2 and flag
  the small ones. Clearing L2 between launches by touching a >100 MB buffer is
  an option for a "cold" number; report cold vs warm separately if you do this.
- For decode, both cards are bandwidth-class-equivalent (~7% apart). Large
  differences between them in a memory-bound kernel point at L2 or at a bug.
- Record `nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv`
  in every run.

Clock lock commands (`scripts/gpu-lock.sh` wraps these):

```
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1695,1695     # 3090; use 2520,2520 for the 4090, or lower if thermals throttle
sudo nvidia-smi -lmc <mem_clk>     # usually "not supported" on GeForce; record the outcome
sudo nvidia-smi -rgc; sudo nvidia-smi -rmc   # unlock
```

If sustained runs hit the power limit at the boost lock, lock lower (e.g. 1500
on the 3090) and recompute the compute roofline. Bandwidth roofline is
unaffected by the graphics clock.
