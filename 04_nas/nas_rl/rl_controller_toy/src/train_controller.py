from __future__ import annotations
import argparse
import random
import torch
import torch.optim as optim

from controller import NASController
from rollout import do_rollout
from log_utils import append_jsonl

def build_space_spec(num_classes: int = 10):
    # Keep consistent with Day 1 space but flattened for controller outputs
    spec = {
        "stem_channels": [8, 16],
        "head_channels": [64, 128],
        "num_classes": [num_classes],
    }
    # 3 stages
    for i in range(3):
        spec[f"stage{i}_depth"] = [1, 2] if i == 0 else [2, 3]
        spec[f"stage{i}_out_ch"] = [16, 24] if i == 0 else ([24, 32, 40] if i == 1 else [40, 64])
        spec[f"stage{i}_block"] = ["conv", "mbconv"] if i == 0 else ["mbconv"]
        spec[f"stage{i}_k"] = [3, 5]
        spec[f"stage{i}_exp"] = [2, 4] if i == 0 else ([2, 4, 6] if i == 1 else [4, 6])
        spec[f"stage{i}_se"] = [0, 1]
    return spec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=str, default="results/rollouts.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = build_space_spec()
    ctrl = NASController(spec)
    opt = optim.Adam(ctrl.parameters(), lr=args.lr)

    baseline = 0.0
    beta = 0.9  # EMA for baseline

    for step in range(args.steps):
        arch, logp, r, comps = do_rollout(ctrl)

        baseline = beta * baseline + (1.0 - beta) * r
        adv = r - baseline

        loss = -(logp * adv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        rec = {
            "seed": args.seed,
            "step": step,
            "reward": float(r),
            "baseline": float(baseline),
            "adv": float(adv),
            "loss": float(loss.item()),
            "arch": arch,
            "components": comps,
        }
        append_jsonl(args.log, rec)

        if (step + 1) % 25 == 0:
            print(f"step {step+1:04d}  reward={r:.4f}  baseline={baseline:.4f}  adv={adv:.4f}")

    print("done. log:", args.log)
    print("greedy arch:", ctrl.greedy())

if __name__ == "__main__":
    main()
