from __future__ import annotations
from typing import Dict[str, float], Tuple
import torch
import math

def gradnorm_score(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    model.train()
    model.zero_grad(set_to_none=True)

    logits = model(X)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    s2 = 0.0
    n = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        s2 += float(torch.sum(g * g).item())
        n += p.numel()

    gn = math.sqrt(s2)
    comps = {
        "gradnorm_l2": float(gn),
        "gradnorm_l2_sq": float(s2),
        "n_params": float(n),
        "loss": float(loss.item()),
    }
    return float(gn), comps
