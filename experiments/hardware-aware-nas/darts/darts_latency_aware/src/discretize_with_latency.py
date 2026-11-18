from __future__ import annotations
import argparse

from utils import write_json
from latency_table import default_latency_table
from latency_regularizer import supernet_expected_latency

from supernet import DARTSSupernet
from discretize import extract_genotype

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/genotype_with_latency.json")
    args = ap.parse_args()

    net = DARTSSupernet(C=16, num_cells=4)
    table = default_latency_table()
    lat, _ = supernet_expected_latency(net, table)
    geno = extract_genotype(net)
    write_json(args.out, {"genotype": geno["genotype"], "expected_latency_total": float(lat.detach().item())})
    print("wrote:", args.out)

if __name__ == "__main__":
    main()
