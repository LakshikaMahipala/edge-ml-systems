from __future__ import annotations
import torch

def make_one_batch(batch: int = 64, num_classes: int = 10, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(batch, 3, 32, 32, generator=g)
    y = torch.randint(0, num_classes, (batch,), generator=g)
    return X, y
