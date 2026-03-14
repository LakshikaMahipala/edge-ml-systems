from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from data_stub import make_loaders
from supernet_model import Supernet
from sampler import sample_subnet
from utils import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    set_seed(args.seed)
    train_loader, _ = make_loaders(batch=args.batch, seed=args.seed)
    it = iter(train_loader)

    kernels = [3, 5]
    widths = [16, 24]
    net = Supernet(widths=widths, kernels=kernels)

    opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=3e-4)

    for step in range(args.steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(train_loader)
            xb, yb = next(it)

        cfg = sample_subnet(kernels, widths)
        net.set_subnet(cfg)

        net.train()
        logits = net(xb)
        loss = F.cross_entropy(logits, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % 25 == 0:
            print(f"step {step+1:04d} loss={loss.item():.4f} cfg={cfg}")

if __name__ == "__main__":
    main()
