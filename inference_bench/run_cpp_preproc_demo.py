# inference_bench/run_cpp_preproc_demo.py
from __future__ import annotations

import argparse
import platform
import sys

import torch

from inference_bench.src.cpp_preproc import CppPreprocConfig, available, load_resize_normalize_to_torch
from inference_bench.src.pytorch_infer import PyTorchConfig, PyTorchRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="Path to an image file")
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    print("C++ Preproc Demo (Week 2 Day 4)")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__}")
    print("")

    if not available():
        print("cpp_preproc_ext not available. Build C++ extension first (cpp_preproc/python/README.md).")
        return

    # 1) C++ preprocess -> torch tensor NCHW
    x = load_resize_normalize_to_torch(args.image, CppPreprocConfig(out_h=args.input_size, out_w=args.input_size, device=args.device))

    # 2) Run inference on that tensor (bypass Python preprocess)
    runner = PyTorchRunner(PyTorchConfig(model_name=args.model, device=args.device, input_size=args.input_size, topk=args.topk, batch=1))
    logits = runner.forward(x)
    pred_idx, pred_prob = runner.postprocess(logits)

    print(f"Output top-{args.topk}: {list(zip(pred_idx, [round(p, 6) for p in pred_prob]))}")


if __name__ == "__main__":
    main()
