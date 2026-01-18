# inference_bench/run_benchmark.py
from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from inference_bench.src.timer import Timer


def preprocess(n: int) -> None:
    # Dummy preprocessing: some math ops
    acc = 0.0
    for i in range(1, n):
        acc += math.sin(i) * 0.000001
    if acc < 0:
        raise RuntimeError("Impossible")


def inference(n: int) -> None:
    # Dummy "inference": heavier math ops
    acc = 0.0
    for i in range(1, n):
        acc += math.sqrt(i)
    if acc < 0:
        raise RuntimeError("Impossible")


def postprocess(n: int) -> None:
    # Dummy postprocess
    s = 0.0
    for i in range(1, n):
        s += (i % 7) * 0.001
    if s < 0:
        raise RuntimeError("Impossible")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--pre_n", type=int, default=10_000)
    ap.add_argument("--infer_n", type=int, default=50_000)
    ap.add_argument("--post_n", type=int, default=10_000)
    args = ap.parse_args()

    timer = Timer(warmup_iters=args.warmup, measure_iters=args.iters)

    # Measure components
    r_pre = timer.run(lambda: preprocess(args.pre_n))
    r_inf = timer.run(lambda: inference(args.infer_n))
    r_post = timer.run(lambda: postprocess(args.post_n))

    # Measure end-to-end pipeline
    def e2e() -> None:
        preprocess(args.pre_n)
        inference(args.infer_n)
        postprocess(args.post_n)

    r_e2e = timer.run(e2e)

    print("Inference Bench (Day 2 baseline)")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Args: warmup={args.warmup}, iters={args.iters}, pre_n={args.pre_n}, infer_n={args.infer_n}, post_n={args.post_n}")
    print("")

    print("Preprocess:", r_pre.summary())
    print("Inference :", r_inf.summary())
    print("Postproc  :", r_post.summary())
    print("")
    print("End-to-end:", r_e2e.summary())
    print("")
    print("Note: End-to-end p50 should be roughly pre+p50 + inf+p50 + post+p50 (not exact, but comparable).")


if __name__ == "__main__":
    main()
