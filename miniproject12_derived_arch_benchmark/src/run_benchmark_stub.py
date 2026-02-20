from __future__ import annotations
import argparse
import torch

from bench_timer import BenchTimer
from baseline_small_cnn import BaselineSmallCNN
from discrete_model_from_genotype import DiscreteDARTSModel, edge_ops_from_genotype_json
from discrete_model_from_subnet import DiscreteOneShotModel, load_subnet_cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genotype", type=str, default="../darts_impl/results/genotype.json")
    ap.add_argument("--subnet", type=str, default="../oneshot_supernet/results/best_subnet.json")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    x = torch.randn(1, 3, 32, 32)

    timer = BenchTimer(warmup=args.warmup, iters=args.iters)

    models = {}

    models["baseline_smallcnn"] = BaselineSmallCNN()
    edge_ops = edge_ops_from_genotype_json(args.genotype)
    models["darts_discrete"] = DiscreteDARTSModel(edge_ops=edge_ops)
    cfg = load_subnet_cfg(args.subnet)
    models["oneshot_discrete"] = DiscreteOneShotModel(cfg)

    for name, m in models.items():
        m.eval()
        with torch.no_grad():
            r = timer.run(lambda: m(x))
        print(name, r)

    print("Run later: write CSV/JSON outputs once running locally.")

if __name__ == "__main__":
    main()
