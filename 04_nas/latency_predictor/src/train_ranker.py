from __future__ import annotations
import argparse
import json
import random
import torch
import torch.optim as optim
from pathlib import Path

from dataset import load_jsonl, train_val_split
from features import graph_to_features, features_to_vector
from model_ranker import RankScorer
from eval_metrics import spearman

def make_pairs(rows, n_pairs: int = 2000, seed: int = 0):
    rng = random.Random(seed)
    pairs = []
    for _ in range(n_pairs):
        a, b = rng.sample(rows, 2)
        la = a["target"]["latency_p50_ms"]
        lb = b["target"]["latency_p50_ms"]
        # label=1 means a is slower than b
        y = 1 if la > lb else 0
        pairs.append((a["graph_id"], b["graph_id"], y))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/ranking_results.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    tr, va = train_val_split(rows, val_ratio=0.2, seed=args.seed)

    # build feature table
    feats = {}
    for r in rows:
        f = graph_to_features(r)
        feats[r["graph_id"]] = f
    keys = sorted(next(iter(feats.values())).keys())

    def vec(gid):
        return features_to_vector(feats[gid], keys)

    pairs_tr = make_pairs(tr, n_pairs=args.pairs, seed=args.seed)
    pairs_va = make_pairs(va, n_pairs=max(500, args.pairs // 5), seed=args.seed + 1)

    X1 = torch.tensor([vec(a) for a,_,_ in pairs_tr], dtype=torch.float32)
    X2 = torch.tensor([vec(b) for _,b,_ in pairs_tr], dtype=torch.float32)
    y  = torch.tensor([lbl for *_,lbl in pairs_tr], dtype=torch.float32)

    torch.manual_seed(args.seed)
    model = RankScorer(d_in=X1.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # pairwise logistic loss: P(a slower than b) = sigmoid(s(a)-s(b))
    for _ in range(args.epochs):
        model.train()
        s1 = model(X1)
        s2 = model(X2)
        logits = s1 - s2
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # evaluate: pair accuracy + graph-level spearman
    model.eval()
    with torch.no_grad():
        # pair accuracy
        V1 = torch.tensor([vec(a) for a,_,_ in pairs_va], dtype=torch.float32)
        V2 = torch.tensor([vec(b) for _,b,_ in pairs_va], dtype=torch.float32)
        vy = torch.tensor([lbl for *_,lbl in pairs_va], dtype=torch.float32)
        p = torch.sigmoid(model(V1) - model(V2))
        pred = (p > 0.5).float()
        pair_acc = float((pred == vy).float().mean().item())

        # spearman over graphs (score vs true latency)
        gids = [r["graph_id"] for r in va]
        scores = [float(model(torch.tensor([vec(g)], dtype=torch.float32)).item()) for g in gids]
        true_lat = [float(r["target"]["latency_p50_ms"]) for r in va]
        sp = spearman(true_lat, scores)  # ranks should align

    out = {
        "n_train": len(tr),
        "n_val": len(va),
        "pairs_train": len(pairs_tr),
        "pairs_val": len(pairs_va),
        "pair_acc": pair_acc,
        "spearman_latency_vs_score": sp,
        "feature_keys": keys,
        "note": "pairwise ranker; good for few-shot latency ranking",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote:", args.out)
    print(out)

if __name__ == "__main__":
    main()
