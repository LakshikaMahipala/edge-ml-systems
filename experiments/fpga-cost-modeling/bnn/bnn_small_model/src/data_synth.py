from __future__ import annotations
import torch

def make_synth_classification(n: int = 20000, in_dim: int = 32, num_classes: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)

    W = torch.randn(num_classes, in_dim, generator=g) * 1.5
    b = torch.randn(num_classes, generator=g) * 0.2
    logits = X @ W.t() + b
    y = torch.argmax(logits + 0.25 * torch.randn_like(logits, generator=g), dim=1)

    X = X + 0.1 * torch.sin(X)
    return X, y
