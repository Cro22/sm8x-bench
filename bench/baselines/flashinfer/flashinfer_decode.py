"""FlashInfer single-request decode-attention driver (run UNDER nsys by
bench/run_attention_flashinfer.py). Generates seeded q/k/v, validates the
FlashInfer output against an fp32 reference, then times PER_BATCH x SAMPLES
launches with CUDA events. Prints the same format the other drivers emit so the
Python runner parses it identically.

GQA: 32 query heads, 8 kv heads, head_dim 128, fp16 KV, batch 1, one query token.
KV is contiguous (single_decode_with_kv_cache) — MAX reads paged KV (page 128);
both read the whole cache once, so the roofline/bandwidth comparison holds. The
access-pattern difference is noted in the report.

Runs in the .venv-attn environment (torch + flashinfer), NOT the max env.

argv: seq_len [seed]
"""
import sys
import torch
import flashinfer

H, HKV, D = 32, 8, 128          # q heads, kv heads, head_dim
WARMUP, SAMPLES, PER_BATCH = 10, 12, 10


def main() -> int:
    seq = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dev = torch.device("cuda")
    torch.manual_seed(seed)

    q = torch.randn(H, D, dtype=torch.float16, device=dev)
    k = torch.randn(seq, HKV, D, dtype=torch.float16, device=dev)
    v = torch.randn(seq, HKV, D, dtype=torch.float16, device=dev)

    # ---- fp32 reference (GQA: q head h attends kv head h // (H//HKV)). ----
    scale = 1.0 / (D ** 0.5)
    qf, kf, vf = q.float(), k.float(), v.float()
    g = H // HKV
    ref = torch.empty(H, D, dtype=torch.float32, device=dev)
    for h in range(H):
        kv = h // g
        s = (qf[h] @ kf[:, kv, :].T) * scale          # [seq]
        a = torch.softmax(s, dim=-1)
        ref[h] = a @ vf[:, kv, :]                       # [D]

    # ---- FlashInfer decode. ----
    def run():
        return flashinfer.single_decode_with_kv_cache(q, k, v)

    out = run().float()
    torch.cuda.synchronize()
    ae = (out - ref).abs()
    max_abs = ae.max().item()
    max_rel = (ae / (ref.abs() + 1e-6)).max().item()
    l2_rel = (ae.pow(2).sum().sqrt() / (ref.pow(2).sum().sqrt() + 1e-12)).item()
    ok = l2_rel < 3e-2
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"correctness: {'PASS' if ok else 'FAIL'} l2_rel_err= {l2_rel:g} "
          f"max_abs_err= {max_abs:g} max_rel_err= {max_rel:g}")
    if not ok:
        print(f"aborting: L2 {l2_rel:g} >= 3e-2")
        return 1

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(SAMPLES):
        torch.cuda.synchronize()
        start.record()
        for _ in range(PER_BATCH):
            run()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop) * 1000.0 / PER_BATCH)  # us/launch

    print(f"launches_per_sample= {PER_BATCH}")
    print("samples_us= " + ",".join(f"{s:g}" for s in samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
