from __future__ import annotations

from inference_bench.src.reporting import RunMeta, PerfSummary, print_budget, save_json

import argparse
import platform
import sys

import numpy as np
import torch

from inference_bench.src.timer import Timer
from inference_bench.src.pytorch_infer import PyTorchConfig, PyTorchRunner
from inference_bench.src.onnxrt_infer import ONNXRTConfig, ONNXRTRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--save_json", action="store_true")
    args = ap.parse_args()

    # Use PyTorchRunner for preprocess/postprocess parity with the PyTorch benchmark.
    pt_cfg = PyTorchConfig(
        model_name=args.model,
        device="cpu",  # keep preprocess on CPU for consistency with ORT CPU
        input_size=args.input_size,
        batch=args.batch,
        topk=args.topk,
    )
    pt_runner = PyTorchRunner(pt_cfg)

    ort_runner = ONNXRTRunner(
        ONNXRTConfig(
            model_name=args.model,
            input_size=args.input_size,
            batch=args.batch,
            providers=("CPUExecutionProvider",),
        )
    )

    timer = Timer(warmup_iters=args.warmup, measure_iters=args.iters, sync=None)

    def do_pre() -> np.ndarray:
        x_t = pt_runner.preprocess()  # torch NCHW float32
        return x_t.detach().cpu().numpy().astype(np.float32)

    def do_inf(x_np: np.ndarray) -> np.ndarray:
        return ort_runner.forward(x_np)

    def do_post(logits_np: np.ndarray):
        # Reuse PyTorchRunner postprocess by converting to torch tensor
        logits_t = torch.from_numpy(logits_np)
        return pt_runner.postprocess(logits_t)

    r_pre = timer.run(lambda: do_pre())
    x = do_pre()

    r_inf = timer.run(lambda: do_inf(x))
    logits = do_inf(x)

    r_post = timer.run(lambda: do_post(logits))
    _ = do_post(logits)

    r_e2e = timer.run(lambda: do_post(do_inf(do_pre())))

    perf = PerfSummary(
        preprocess=r_pre.summary(),
        inference=r_inf.summary(),
        postprocess=r_post.summary(),
        end_to_end=r_e2e.summary(),
    )

    print("ONNX Runtime Benchmark (Week 3 Day 6 — Mini-project 1B)")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__}")
    print(f"Args: model={args.model}, input_size={args.input_size}, batch={args.batch}, warmup={args.warmup}, iters={args.iters}, topk={args.topk}")
    print("")
    print_budget(perf)

    meta = RunMeta(
        project="inference_bench",
        script="run_onnxruntime_benchmark.py",
        model=args.model,
        device="onnxruntime_cpu",
        input_desc=f"{args.batch}x3x{args.input_size}x{args.input_size}",
        warmup=args.warmup,
        iters=args.iters,
        platform=platform.platform(),
        python=sys.version.split()[0],
        torch=torch.__version__,
    )

    if args.save_json:
        out_path = save_json(meta, perf)
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
