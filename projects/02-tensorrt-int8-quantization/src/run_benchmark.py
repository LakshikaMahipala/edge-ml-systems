"""PyTorch -> ONNX -> TensorRT FP32 / FP16 / INT8: accuracy and latency on one GPU.

For each model this script:
  1. exports the ImageNet-pretrained torchvision model to ONNX (dynamic batch),
  2. builds FP32, FP16 and INT8 TensorRT engines (INT8 calibrated on Imagenette *train*),
  3. measures top-1 accuracy on Imagenette *val* (3,925 images),
  4. measures GPU latency (p50 / p99) at batch 1 and throughput at batch 32,
and compares everything to PyTorch FP32 eager mode.

Example:
    python src/run_benchmark.py --models resnet18 mobilenet_v3_large --out results/t4
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

from data import download_imagenette, make_loaders
from trt_engine import TorchEntropyCalibrator, TrtRunner, build_engine


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, x, str(path), opset_version=17, input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})


@torch.no_grad()
def top1(predict, loader, class_idx: torch.Tensor) -> float:
    """Top-1 over the 10 Imagenette classes (argmax restricted to their logits)."""
    correct = total = 0
    class_idx = class_idx.cuda()
    for x, y in loader:
        logits = predict(x.cuda())
        pred = logits[:, class_idx].argmax(dim=1).cpu()
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / total


@torch.no_grad()
def torch_latency_ms(model, x, warmup=20, iters=200) -> list[float]:
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        model(x)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return times


def summarize(times_b1: list[float], times_b32: list[float]) -> dict:
    return {
        "p50_ms_b1": float(np.percentile(times_b1, 50)),
        "p99_ms_b1": float(np.percentile(times_b1, 99)),
        "throughput_img_s_b32": 32 * 1000.0 / float(np.median(times_b32)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "mobilenet_v3_large"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/t4")
    ap.add_argument("--n_calib", type=int, default=512, help="calibration images (from train split)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs an NVIDIA GPU"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = download_imagenette(args.data)
    gpu = torch.cuda.get_device_name()
    rows = []

    for name in args.models:
        print(f"\n=== {name} on {gpu} ===")
        model = torchvision.models.get_model(name, weights="DEFAULT").eval()
        onnx_path = out / f"{name}.onnx"
        export_onnx(model, onnx_path)
        model = model.cuda()

        calib_loader, eval_loader, class_idx = make_loaders(root, args.batch, args.n_calib)
        x1 = torch.randn(1, 3, 224, 224, device="cuda")
        x32 = torch.randn(args.batch, 3, 224, 224, device="cuda")

        # Baseline: PyTorch FP32 eager
        acc = top1(model, eval_loader, class_idx)
        rows.append({"model": name, "backend": "pytorch", "precision": "fp32", "top1": acc,
                     "build_s": 0.0, "engine_mb": 0.0,
                     **summarize(torch_latency_ms(model, x1), torch_latency_ms(model, x32))})
        print(rows[-1])

        for precision in ["fp32", "fp16", "int8"]:
            calibrator = None
            if precision == "int8":
                calibrator = TorchEntropyCalibrator((x for x, _ in calib_loader), args.batch,
                                                    out / f"{name}_int8.cache")
            t0 = time.time()
            engine = build_engine(onnx_path, precision, max_batch=args.batch, calibrator=calibrator)
            build_s = time.time() - t0
            (out / f"{name}_{precision}.plan").write_bytes(engine)

            runner = TrtRunner(engine)
            acc = top1(runner, eval_loader, class_idx)
            rows.append({"model": name, "backend": "tensorrt", "precision": precision, "top1": acc,
                         "build_s": build_s, "engine_mb": len(engine) / 1e6,
                         **summarize(runner.latency_ms(x1), runner.latency_ms(x32))})
            print(rows[-1])

    csv_path = out / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    (out / "environment.txt").write_text(
        f"gpu={gpu}\ntorch={torch.__version__}\ntorchvision={torchvision.__version__}\n"
        f"tensorrt={__import__('tensorrt').__version__}\ncuda={torch.version.cuda}\n")
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
