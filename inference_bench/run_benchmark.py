from __future__ import annotations

import argparse
import torch

from week01_inference_bench.src.infer import build_model, default_hooks, resolve_device
from week01_inference_bench.src.timer import benchmark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week01 baseline PyTorch inference benchmark (Day 3)")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--model", default="resnet18", choices=["resnet18"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-batches", type=int, default=10)
    p.add_argument("--measure-batches", type=int, default=50)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    model = build_model(args.model, pretrained=bool(args.pretrained)).to(device)
    hooks = default_hooks()

    # Day 3: synthetic batch so we validate the pipeline + measurement first.
    batch = torch.randn(args.batch_size, 3, 224, 224, device=device)

    sync_fn = torch.cuda.synchronize if device.type == "cuda" else None

    @torch.no_grad()
    def step(i: int) -> int:
        hooks.before_batch(i)

        # Preprocess (minimal Day3): already correct shape on device
        x = batch

        # Model forward
        logits = model(x)

        # Postprocess (minimal): keep logits
        hooks.after_batch(i, logits)

        return int(x.shape[0])

    r = benchmark(step, warmup=args.warmup_batches, iters=args.measure_batches, synchronize_fn=sync_fn)

    print("=== Week01 Day3: PyTorch baseline ===")
    print(f"device: {device}")
    print(r.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
