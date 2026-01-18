# inference_bench/run_profile_pytorch.py
from __future__ import annotations

import argparse
import platform
import sys

import torch
from torch.profiler import profile, ProfilerActivity

from inference_bench.src.pytorch_infer import PyTorchConfig, PyTorchRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    cfg = PyTorchConfig(model_name=args.model, device=args.device, input_size=args.input_size)
    runner = PyTorchRunner(cfg)

    activities = [ProfilerActivity.CPU]
    if args.device.startswith("cuda") and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print("PyTorch Profiler Run (Day 4)")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__}")
    print(f"Args: model={args.model}, device={args.device}, input_size={args.input_size}, warmup={args.warmup}, steps={args.steps}")
    print("")

    # Warmup
    for _ in range(args.warmup):
        runner.end_to_end()

    # Profile a short run (keep it small)
    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(args.steps):
            runner.end_to_end()

    # Print top ops by self time
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))


if __name__ == "__main__":
    main()
