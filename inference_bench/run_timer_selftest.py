# inference_bench/run_timer_selftest.py
from __future__ import annotations

import math
import platform
import sys
from inference_bench.src.timer import Timer


def cpu_work(n: int = 50_000) -> None:
    # Deterministic CPU workload (no prints inside!)
    acc = 0.0
    for i in range(1, n):
        acc += math.sqrt(i)
    # Prevent potential over-optimization (very minor guard)
    if acc < 0:
        raise RuntimeError("Impossible")


def main() -> None:
    timer = Timer(warmup_iters=20, measure_iters=200)

    # Run two workloads to see scaling
    r1 = timer.run(lambda: cpu_work(20_000))
    r2 = timer.run(lambda: cpu_work(80_000))

    print("Timer self-test")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print("")
    print("Workload A (n=20k)")
    print(r1.summary())
    print("")
    print("Workload B (n=80k)")
    print(r2.summary())
    print("")
    print("Expected: Workload B should be noticeably slower than A.")


if __name__ == "__main__":
    main()
