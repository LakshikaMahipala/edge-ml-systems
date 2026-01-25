# inference_bench/src/io_bench.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class IOResult:
    bytes_total: int
    iters: int
    p50_ms: float
    p99_ms: float
    mean_ms: float
    throughput_mb_s: float

    def to_dict(self):
        return {
            "bytes_total": self.bytes_total,
            "iters": self.iters,
            "p50_ms": self.p50_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "throughput_mb_s": self.throughput_mb_s,
        }


def percentile_ms(samples_s: List[float], p: float) -> float:
    arr = np.array(samples_s, dtype=np.float64) * 1000.0
    return float(np.percentile(arr, p))


def read_file_once(path: Path, chunk_bytes: int) -> int:
    """
    Read file in chunks and return total bytes read.
    chunk_bytes controls syscalls and buffer behavior.
    """
    total = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            total += len(b)
    return total


def try_drop_caches_linux() -> bool:
    """
    Best-effort attempt to drop Linux page cache.
    Requires root privileges:
      sync; echo 3 > /proc/sys/vm/drop_caches
    We do not assume permissions; returns False if not possible.
    """
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except Exception:
        return False


def bench_file_read(
    path: Path,
    iters: int,
    chunk_bytes: int,
    drop_caches: bool = False,
) -> IOResult:
    times: List[float] = []
    bytes_total = 0

    for _ in range(iters):
        if drop_caches:
            try_drop_caches_linux()

        t0 = time.perf_counter()
        n = read_file_once(path, chunk_bytes=chunk_bytes)
        t1 = time.perf_counter()

        dt = t1 - t0
        times.append(dt)
        bytes_total = n  # same each iter

    mean_s = float(np.mean(times))
    p50_ms = percentile_ms(times, 50)
    p99_ms = percentile_ms(times, 99)
    mean_ms = mean_s * 1000.0

    # Throughput computed using mean time (stable aggregate)
    throughput_mb_s = (bytes_total / (1024.0 * 1024.0)) / max(mean_s, 1e-12)

    return IOResult(
        bytes_total=bytes_total,
        iters=iters,
        p50_ms=p50_ms,
        p99_ms=p99_ms,
        mean_ms=mean_ms,
        throughput_mb_s=float(throughput_mb_s),
    )
