from __future__ import annotations
import time
import torch

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())

def train_epoch(model, opt, X, y, batch: int = 256):
    model.train()
    n = X.shape[0]
    idx = torch.randperm(n)
    total_loss = 0.0
    for i in range(0, n, batch):
        j = idx[i:i+batch]
        xb, yb = X[j], y[j]
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * xb.shape[0]
    return total_loss / n

@torch.no_grad()
def eval_model(model, X, y, batch: int = 512):
    model.eval()
    n = X.shape[0]
    acc_sum = 0.0
    for i in range(0, n, batch):
        xb, yb = X[i:i+batch], y[i:i+batch]
        logits = model(xb)
        acc_sum += accuracy(logits, yb) * xb.shape[0]
    return acc_sum / n

def wall_ms(fn, iters: int = 300, warmup: int = 100):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters
