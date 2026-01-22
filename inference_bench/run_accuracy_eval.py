# inference_bench/run_accuracy_eval.py
from __future__ import annotations

import argparse
import platform
import sys

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from inference_bench.src.imagenet_eval import evaluate_classifier


def load_model(name: str) -> torch.nn.Module:
    import torchvision.models as models
    if name == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if name == "mobilenet_v3_small":
        return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    if name == "efficientnet_b0":
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    raise ValueError(f"Unsupported model: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=50, help="limit evaluation for quick runs; set -1 for full test set")
    args = ap.parse_args()

    device = torch.device(args.device)

    # CIFAR-10 images are 32x32; our pretrained ImageNet models expect 224x224.
    # We resize and normalize using ImageNet stats. This is NOT perfect accuracy,
    # but is excellent for teaching and for consistent benchmarking methodology.
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))

    model = load_model(args.model).to(device).eval()

    max_batches = None if args.max_batches < 0 else args.max_batches
    acc = evaluate_classifier(model=model, dataloader=dl, device=device, max_batches=max_batches)

    print("Accuracy Eval (Day 6 Mini-Project 0)")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__}")
    print(f"Args: model={args.model}, device={args.device}, batch={args.batch}, num_workers={args.num_workers}, max_batches={args.max_batches}")
    print("")
    print({"top1": acc.top1, "top5": acc.top5, "n": acc.n})
    print("")
    print("Note:")
    print("- This uses ImageNet-pretrained models evaluated on resized CIFAR-10.")
    print("- The goal is to build a correct evaluation pipeline (top-1/top-5), not to maximize CIFAR-10 accuracy.")
    print("- Later, we can switch to ImageNet/Imagenette for more meaningful accuracy.")
    

if __name__ == "__main__":
    main()
