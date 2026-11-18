from __future__ import annotations
from typing import Dict, Any, Tuple
import torch

def snip_like_score(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    model.train()
    model.zero_grad(set_to_none=True)

    logits = model(X)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    total = 0.0
    total_absgrad = 0.0
    total_absw = 0.0
    n = 0

    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        w = p.detach()
        total += float(torch.sum(torch.abs(w * g)).item())
        total_absgrad += float(torch.sum(torch.abs(g)).item())
        total_absw += float(torch.sum(torch.abs(w)).item())
        n += p.numel()

    comps = {
        "snip_sum_abs_wg": float(total),
        "sum_abs_grad": float(total_absgrad),
        "sum_abs_w": float(total_absw),
        "n_params": float(n),
        "loss": float(loss.item()),
    }
    return float(total), comps
