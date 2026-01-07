import time
from inference_bench.src.timer import benchmark


def step(_i: int) -> int:
    time.sleep(0.002)  # ~2ms work
    return 32          # pretend "32 items per batch"


if __name__ == "__main__":
    r = benchmark(step, warmup=5, iters=50)
    print(r.summary())
