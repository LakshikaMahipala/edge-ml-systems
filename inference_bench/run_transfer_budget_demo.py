# inference_bench/run_transfer_budget_demo.py
from __future__ import annotations

import argparse

from inference_bench.src.transfer_model import (
    PCIE3_X16,
    PCIE4_X16,
    SHARED_MEM,
    bytes_for_tensor,
    estimate_transfer_ms,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--c", type=int, default=3)
    ap.add_argument("--h", type=int, default=224)
    ap.add_argument("--w", type=int, default=224)
    ap.add_argument("--dtype_bytes", type=int, default=4, help="4 for fp32, 2 for fp16, 1 for int8")
    args = ap.parse_args()

    num_bytes = bytes_for_tensor(args.n, args.c, args.h, args.w, dtype_bytes=args.dtype_bytes)
    mb = num_bytes / (1024.0 * 1024.0)

    print("Transfer Budget Demo (Week 3 Day 4)")
    print(f"Tensor: {args.n}x{args.c}x{args.h}x{args.w}, dtype_bytes={args.dtype_bytes} -> {mb:.3f} MB")
    print("")

    for link in [SHARED_MEM, PCIE3_X16, PCIE4_X16]:
        t_ms = estimate_transfer_ms(num_bytes, link)
        print(f"{link.name:24s}: {t_ms:.4f} ms (bandwidth={link.bandwidth_gbps} GB/s, overhead={link.overhead_us} us)")

    print("")
    print("Interpretation:")
    print("- Small tensors are dominated by overhead, not bandwidth.")
    print("- Batch=1 workloads often suffer from overhead-dominated transfers.")
    print("- This is why pipelining, batching, and pinned memory matter on GPU/accelerators.")


if __name__ == "__main__":
    main()
