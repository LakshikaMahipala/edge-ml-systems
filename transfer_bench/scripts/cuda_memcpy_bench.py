import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch


def percentile(xs, p):
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(xs, p))


def bench_copy(fn, warmup, iters, sync):
    # warmup
    for _ in range(warmup):
        fn()
    sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)  # ms
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": percentile(times, 50),
        "p99_ms": percentile(times, 99),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--sizes_mb", type=str, default="1,4,16,64,256")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", type=str, default="results/memcpy_bench.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run this later on a GPU machine.")

    dev = torch.device(args.device)
    sizes_mb = [int(s.strip()) for s in args.sizes_mb.split(",") if s.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def sync():
        torch.cuda.synchronize()

    results = {
        "meta": {
            "device": args.device,
            "warmup": args.warmup,
            "iters": args.iters,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "runs": []
    }

    for mb in sizes_mb:
        n_bytes = mb * 1024 * 1024
        n_f32 = n_bytes // 4

        # Pageable host tensor
        h_page = torch.empty(n_f32, dtype=torch.float32, device="cpu", pin_memory=False)
        # Pinned host tensor
        h_pin = torch.empty(n_f32, dtype=torch.float32, device="cpu", pin_memory=True)

        d = torch.empty_like(h_page, device=dev)

        # Copy ops (measure only transfer, no extra compute)
        r_h2d_page = bench_copy(lambda: d.copy_(h_page, non_blocking=False), args.warmup, args.iters, sync)
        r_h2d_pin  = bench_copy(lambda: d.copy_(h_pin,  non_blocking=True),  args.warmup, args.iters, sync)

        r_d2h_page = bench_copy(lambda: h_page.copy_(d, non_blocking=False), args.warmup, args.iters, sync)
        r_d2h_pin  = bench_copy(lambda: h_pin.copy_(d,  non_blocking=True),  args.warmup, args.iters, sync)

        def bw_gbps(stats):
            # bandwidth = bytes / time
            # use p50 for reporting
            t_s = stats["p50_ms"] / 1e3
            if t_s <= 0:
                return None
            return float((n_bytes / t_s) / 1e9)

        results["runs"].append({
            "size_mb": mb,
            "bytes": n_bytes,
            "h2d_pageable": {**r_h2d_page, "bw_gbps_p50": bw_gbps(r_h2d_page)},
            "h2d_pinned":   {**r_h2d_pin,  "bw_gbps_p50": bw_gbps(r_h2d_pin)},
            "d2h_pageable": {**r_d2h_page, "bw_gbps_p50": bw_gbps(r_d2h_page)},
            "d2h_pinned":   {**r_d2h_pin,  "bw_gbps_p50": bw_gbps(r_d2h_pin)},
        })

        print(f"[{mb} MB] done")

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
