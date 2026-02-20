from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import set_seed, append_jsonl, write_json
from latency_table import default_latency_table
from latency_regularizer import supernet_expected_latency

# reuse DARTS supernet + data loader from darts_impl
from data_stub import make_loaders
from supernet import DARTSSupernet
from discretize import extract_genotype

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr_w", type=float, default=0.05)
    ap.add_argument("--lr_a", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--out", type=str, default="results/alpha_latency_history.jsonl")
    ap.add_argument("--out_best", type=str, default="results/best_genotype_latency.json")
    args = ap.parse_args()

    set_seed(args.seed)
    train_loader, val_loader = make_loaders(batch=args.batch, seed=args.seed)
    train_it = iter(train_loader)
    val_it = iter(val_loader)

    net = DARTSSupernet(C=16, num_cells=4)
    opt_w = optim.SGD(net.weight_parameters(), lr=args.lr_w, momentum=0.9, weight_decay=3e-4)
    opt_a = optim.Adam(net.arch_parameters(), lr=args.lr_a, betas=(0.5, 0.999), weight_decay=1e-3)

    table = default_latency_table()
    best = {"score": -1e9, "step": 0, "geno": None, "lat_total": None}

    for step in range(args.steps):
        # ---- w-step (train) ----
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

        # ---- alpha-step (val + latency) ----
        try:
            xv, yv = next(val_it)
        except StopIteration:
            val_it = iter(val_loader)
            xv, yv = next(val_it)

        net.train()
        logits_v = net(xv)
        val_loss = F.cross_entropy(logits_v, yv)
        lat_t, lat_comps = supernet_expected_latency(net, table)
        loss_a = val_loss + args.lam * lat_t

        opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        opt_a.step()

        if (step + 1) % 10 == 0:
            geno = extract_genotype(net)
            # define a simple combined score for tracking (higher is better)
            score = float((-val_loss.detach() - args.lam * lat_t.detach()).item())
            rec = {
                "step": step + 1,
                "loss_w": float(loss_w.item()),
                "val_loss": float(val_loss.item()),
                "lat_total": float(lat_t.detach().item()),
                "lam": float(args.lam),
                "alpha_objective": float(loss_a.item()),
                "track_score": score,
                "lat_edges": lat_comps,
                "genotype": geno["genotype"],
            }
            append_jsonl(args.out, rec)

            if score > best["score"]:
                best = {"score": score, "step": step + 1, "geno": geno, "lat_total": float(lat_t.detach().item())}

            print(f"step {step+1:04d} val_loss={val_loss.item():.4f} lat={lat_t.item():.3f} loss_a={loss_a.item():.4f}")

    write_json(args.out_best, best)
    print("wrote:", args.out, "and", args.out_best)

if __name__ == "__main__":
    main()
