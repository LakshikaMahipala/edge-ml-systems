from __future__ import annotations
from typing import Dict, Any, Tuple
import random
import torch
import torch.optim as optim

from scoring import ProxyScore

# run later with PYTHONPATH=../nas_foundations/tiny_cnn_search_space/src
from model_builder import build_model

def make_toy_image_data(n: int = 2048, num_classes: int = 10, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return X, y

@torch.no_grad()
def eval_acc(model, X, y, batch=256) -> float:
    model.eval()
    correct = 0
    total = 0
    for i in range(0, X.shape[0], batch):
        xb = X[i:i+batch]
        yb = y[i:i+batch]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / total

def train_proxy_once(arch_id: str, arch: Dict[str, Any], steps: int, seed: int = 0) -> ProxyScore:
    random.seed(seed)
    torch.manual_seed(seed)

    model = build_model(arch)
    opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    X, y = make_toy_image_data(n=2048, num_classes=arch["num_classes"], seed=seed)
    Xtr, ytr = X[:1536], y[:1536]
    Xva, yva = X[1536:], y[1536:]

    model.train()
    loss_val = 0.0
    bs = 128
    for s in range(steps):
        j = (s * bs) % Xtr.shape[0]
        xb = Xtr[j:j+bs]
        yb = ytr[j:j+bs]
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        loss_val = float(loss.item())

    acc = float(eval_acc(model, Xva, yva))
    # proxy score: val_acc minus tiny penalty for instability (here just loss)
    score = acc - 0.01 * loss_val

    return ProxyScore(
        arch_id=str(arch_id),
        score=float(score),
        val_acc=float(acc),
        train_loss=float(loss_val),
        steps=int(steps),
        seed=int(seed),
        notes="toy data proxy; replace with real dataset later"
    )
