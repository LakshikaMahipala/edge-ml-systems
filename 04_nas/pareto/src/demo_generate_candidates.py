from __future__ import annotations
import random
from metrics_schema import Candidate

def main():
    random.seed(0)
    cands = []
    for i in range(30):
        # fake tradeoff: faster models slightly worse acc
        latency = random.uniform(1.0, 20.0)
        acc = max(0.0, min(1.0, 0.92 - 0.01 * (latency/2) + random.uniform(-0.02, 0.02)))
        energy = latency * random.uniform(0.4, 0.8)  # proxy
        cands.append(Candidate(f"M{i:02d}", acc, latency, energy))
    for c in cands[:5]:
        print(c)

if __name__ == "__main__":
    main()
