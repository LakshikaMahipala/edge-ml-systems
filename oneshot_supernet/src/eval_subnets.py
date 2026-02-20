from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F

from data_stub import make_loaders
from supernet_model import Supernet
from sampler import sample_subnet
from utils import set_seed, append_jsonl, write_json

@torch.no_grad()
def eval_acc(net, loader):
    net.eval()
    c = 0
    t = 0
    for xb, yb in loader:
        logits = net(xb)
        pred = logits.argmax(dim=1)
        c += int((pred == yb).sum().item())
        t += int(yb.numel())
    return c / t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=str, default="results/subnet_scores.jsonl")
    ap.add_argument("--out_best", type=str, default="results/best_subnet.json")
    args = ap.parse_args()

    set_seed(args.seed)
    _, val_loader = make_loaders(batch=args.batch, seed=args.seed)

    kernels = [3, 5]
    widths = [16, 24]
    net = Supernet(widths=widths, kernels=kernels)

    best = {"acc": -1.0, "cfg": None}
    for i in range(args.n):
        cfg = sample_subnet(kernels, widths, seed=args.seed + i)
        net.set_subnet(cfg)
        acc = float(eval_acc(net, val_loader))
        rec = {"i": i, "acc": acc, "cfg": cfg}
        append_jsonl(args.log, rec)
        if acc > best["acc"]:
            best = {"acc": acc, "cfg": cfg}

        if (i + 1) % 20 == 0:
            print("eval", i + 1, "best_acc", best["acc"])

    write_json(args.out_best, best)
    print("best:", best, "wrote:", args.out_best)

if __name__ == "__main__":
    main()
