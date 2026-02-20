from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from data_stub import make_loaders
from supernet import DARTSSupernet
from discretize import extract_genotype
from utils import set_seed, append_jsonl, write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--cells", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr_w", type=float, default=0.05)
    ap.add_argument("--lr_a", type=float, default=0.01)
    ap.add_argument("--out_alpha", type=str, default="results/alpha_history.jsonl")
    ap.add_argument("--out_geno", type=str, default="results/genotype.json")
    args = ap.parse_args()

    set_seed(args.seed)
    train_loader, val_loader = make_loaders(batch=args.batch, seed=args.seed)
    train_it = iter(train_loader)
    val_it = iter(val_loader)

    net = DARTSSupernet(C=args.C, num_cells=args.cells)
    opt_w = optim.SGD(net.weight_parameters(), lr=args.lr_w, momentum=0.9, weight_decay=3e-4)
    opt_a = optim.Adam(net.arch_parameters(), lr=args.lr_a, betas=(0.5, 0.999), weight_decay=1e-3)

    for step in range(args.steps):
        # ===== w-step (train) =====
        try:
            xb, yb = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            xb, yb = next(train_it)

        net.train()
        logits = net(xb)
        loss_w = F.cross_entropy(logits, yb)

        opt_w.zero_grad(set_to_none=True)
        loss_w.backward()
        opt_w.step()

        # ===== alpha-step (val) =====
        try:
            xv, yv = next(val_it)
        except StopIteration:
            val_it = iter(val_loader)
            xv, yv = next(val_it)

        logits_v = net(xv)
        loss_a = F.cross_entropy(logits_v, yv)

        opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        opt_a.step()

        if (step + 1) % 10 == 0:
            geno = extract_genotype(net)
            rec = {
                "step": step + 1,
                "loss_w": float(loss_w.item()),
                "loss_a": float(loss_a.item()),
                "genotype": geno["genotype"],
            }
            append_jsonl(args.out_alpha, rec)
            print(f"step {step+1:04d}  loss_w={loss_w.item():.4f}  loss_a={loss_a.item():.4f}")

    write_json(args.out_geno, extract_genotype(net))
    print("wrote:", args.out_alpha, "and", args.out_geno)

if __name__ == "__main__":
    main()
