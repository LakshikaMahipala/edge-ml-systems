# inference_bench/run_file_io_benchmark.py
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from inference_bench.src.io_bench import bench_file_read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to file to read (e.g., a video or large image)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--chunk_kb", type=int, default=256, help="Read chunk size in KB")
    ap.add_argument("--drop_caches", action="store_true", help="Try to drop Linux page cache (requires permissions)")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    chunk_bytes = args.chunk_kb * 1024

    res = bench_file_read(
        path=p,
        iters=args.iters,
        chunk_bytes=chunk_bytes,
        drop_caches=args.drop_caches,
    )

    print("File I/O Benchmark (Week 3 Day 1)")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"File: {p} ({res.bytes_total / (1024*1024):.2f} MB)")
    print(f"Args: iters={args.iters}, chunk_kb={args.chunk_kb}, drop_caches={args.drop_caches}")
    print("")
    print(f"p50 read latency: {res.p50_ms:.3f} ms")
    print(f"p99 read latency: {res.p99_ms:.3f} ms")
    print(f"mean read latency: {res.mean_ms:.3f} ms")
    print(f"throughput: {res.throughput_mb_s:.2f} MB/s")
    print("")
    print("Interpretation tips:")
    print("- If drop_caches=False, later iterations may be faster due to page cache.")
    print("- p99 >> p50 indicates tail latency risk for real-time pipelines.")
    print("- Try different chunk sizes to see syscall overhead vs throughput effects.")


if __name__ == "__main__":
    main()
