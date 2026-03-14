# inference_bench/src/imagenet_eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class AccuracyResult:
    top1: float
    top5: float
    n: int

    def to_dict(self) -> Dict[str, float]:
        return {"top1": self.top1, "top5": self.top5, "n": float(self.n)}


@torch.inference_mode()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[int, int]:
    """
    Returns number of correct predictions for each k in topk.
    logits: [B, C]
    targets: [B]
    """
    maxk = max(topk)
    _, pred = torch.topk(logits, k=maxk, dim=1)          # [B, maxk]
    pred = pred.t()                                      # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    out: Dict[int, int] = {}
    for k in topk:
        out[k] = int(correct[:k].reshape(-1).float().sum().item())
    return out


@torch.inference_mode()
def evaluate_classifier(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
) -> AccuracyResult:
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0

    for bi, (x, y) in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        c = topk_accuracy(logits, y, topk=(1, 5))
        bsz = x.shape[0]
        total += bsz
        correct1 += c[1]
        correct5 += c[5]

    top1 = correct1 / max(total, 1)
    top5 = correct5 / max(total, 1)
    return AccuracyResult(top1=top1, top5=top5, n=total)
