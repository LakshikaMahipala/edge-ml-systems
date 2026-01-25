# inference_bench/run_preproc_speedup.py
from __future__ import annotations

from inference_bench.src.reporting import RunMeta, PerfSummary, print_budget, save_json

import argparse
import platform
import sys
import time

import torch

from inference_bench.src.timer import Timer
from inference_bench.src.pytorch_infer import PyTorchConfig, PyTorchRunner
from inference_bench.src.preproc_paths import PreprocPathConfig, python_preprocess_fn, cpp_preprocess_fn
from inference_bench.src.cpp_preproc import available as cpp_available


def measure_path(timer: Timer, pre_fn, runner: PyTorchRunner):
    """
    Measures preprocess, inference, postprocess, end-to-end for a path.
    pre_fn: callable -> tensor
    """
    # preprocess
    r_pre = timer.run(lambda: pre_fn())

    x = pre_fn()
    r_inf = timer.run(lambda: runner.forward(x))

    logits = runner.forward(x)
    r_post = timer.run(lambda: runner.postprocess(logits))

    r_e2e = timer.run(lambda: (runner.postprocess(runner.forward(pre_fn()))))

    perf = PerfSummary(
        preprocess=r_pre.summary(),
        inference=r_inf.summary(),
        postprocess=r_post.summary(),
        end_to_end=r_e2e.summary(),
    )
    return perf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--image", type=str, default="", help="Image path required for C++ preprocess path")
    ap.add_argument("--save_json", action="store_true")
    args = ap.parse_args()

    cfg = PyTorchConfig(
        model_name=args.model,
        device=args.device,
        input_size=args.input_size,
        topk=args.topk,
        batch=args.batch,
    )
    runner = PyTorchRunner(cfg)

    sync = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        sync = torch.cuda.synchronize

    timer = Timer(warmup_iters=args.warmup, measure_iters=args.iters, sync=sync)

    print("Mini-Project 1A (Week 2 Day 6): Python vs C++ Preprocessing")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__}")
    print(f"Args: model={args.model}, device={args.device}, input_size={args.input_size}, batch={args.batch}, warmup={args.warmup}, iters={args.iters}, topk={args.topk}")
    print("")

    # Path A: Python preprocessing
    py_pre = python_preprocess_fn(runner)
    py_perf = measure_path(timer, py_pre, runner)
    print("=== Path A: Python preprocess ===")
    print_budget(py_perf)
    print("")

    # Path B: C++ preprocessing (requires image)
    if args.image:
        if not cpp_available():
            print("C++ preprocess requested but cpp_preproc_ext is not available.")
            print("Build the extension in cpp_preproc/ and ensure it is importable.")
        else:
            cpp_pre = cpp_preprocess_fn(args.image, PreprocPathConfig(input_size=args.input_size, device=args.device))
            cpp_perf = measure_path(timer, cpp_pre, runner)
            print("=== Path B: C++ preprocess ===")
            print_budget(cpp_perf)
            print("")

            # Simple speedup reporting (p50-based)
            py_p50 = py_perf.preprocess.get("p50_ms", None)
            cpp_p50 = cpp_perf.preprocess.get("p50_ms", None)
            if py_p50 and cpp_p50 and cpp_p50 > 0:
                speedup = py_p50 / cpp_p50
                print(f"Preprocess p50 speedup (Python/C++): {speedup:.3f}x")
            print("")

            if args.save_json:
                meta = RunMeta(
                    project="inference_bench",
                    script="run_preproc_speedup.py",
                    model=args.model,
                    device=args.device,
                    input_desc=f"{args.batch}x3x{args.input_size}x{args.input_size}",
                    warmup=args.warmup,
                    iters=args.iters,
                    platform=platform.platform(),
                    python=sys.version.split()[0],
                    torch=torch.__version__,
                )

                out_a = save_json(meta, py_perf)
                print(f"Saved JSON (Python path): {out_a}")

                out_b = save_json(meta, cpp_perf)
                print(f"Saved JSON (C++ path): {out_b}")
    else:
        print("Note: C++ path not run because --image was not provided.")
        print("When you can run locally, pass: --image path/to/image.jpg --save_json")

if __name__ == "__main__":
    main()
