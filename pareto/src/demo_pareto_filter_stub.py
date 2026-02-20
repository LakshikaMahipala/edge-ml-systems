from __future__ import annotations
import random
from metrics_schema import Candidate
from pareto import pareto_front

def main():
    random.seed(0)
    cands = []
    for i in range(30):
        latency = random.uniform(1.0, 20.0)
        acc = max(0.0, min(1.0, 0.92 - 0.01 * (latency/2) + random.uniform(-0.02, 0.02)))
        energy = latency * random.uniform(0.4, 0.8)
        cands.append(Candidate(f"M{i:02d}", acc, latency, energy))

    front = pareto_front(cands)
    print("Total candidates:", len(cands))
    print("Pareto front size:", len(front))
    for c in front[:10]:
        print(c)

    print("Run later: export CSV + plot scatter after local runtime is available.")

if __name__ == "__main__":
    main()
