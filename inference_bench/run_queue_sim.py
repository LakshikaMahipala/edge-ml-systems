# inference_bench/run_queue_sim.py
from __future__ import annotations

import argparse
import random
import statistics


def simulate_mm1(arrival_rate: float, service_rate: float, n: int, seed: int = 0):
    """
    Simple M/M/1 queue simulation.
    - arrival_rate: lambda (jobs/sec)
    - service_rate: mu (jobs/sec)
    If lambda approaches mu, queue grows and tail latency explodes.
    """
    random.seed(seed)

    t = 0.0
    server_free = 0.0
    latencies = []

    for _ in range(n):
        # Exponential inter-arrival
        t += random.expovariate(arrival_rate)
        arrival_time = t

        # Exponential service time
        service_time = random.expovariate(service_rate)

        start_time = max(arrival_time, server_free)
        finish_time = start_time + service_time
        server_free = finish_time

        latency = finish_time - arrival_time
        latencies.append(latency)

    latencies.sort()
    def pct(p):
        idx = int((p / 100.0) * (len(latencies) - 1))
        return latencies[idx]

    return {
        "p50_s": pct(50),
        "p90_s": pct(90),
        "p99_s": pct(99),
        "mean_s": statistics.mean(latencies),
        "utilization": arrival_rate / service_rate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service_ms", type=float, default=20.0, help="mean service time in ms")
    ap.add_argument("--arrival_rps", type=float, default=30.0, help="arrival rate in requests/sec")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mean_service_s = args.service_ms / 1000.0
    service_rate = 1.0 / mean_service_s
    arrival_rate = args.arrival_rps

    out = simulate_mm1(arrival_rate=arrival_rate, service_rate=service_rate, n=args.n, seed=args.seed)
    print("M/M/1 Queue Simulation (Day 5 teaching tool)")
    print(f"Mean service time: {args.service_ms} ms -> service_rate(mu)={service_rate:.2f} jobs/sec")
    print(f"Arrival rate (lambda): {arrival_rate:.2f} req/sec")
    print(f"Utilization rho=lambda/mu: {out['utilization']:.3f}")
    print("")
    print(out)
    print("")
    print("Interpretation:")
    print("- As utilization approaches 1.0, p99 latency grows sharply (queueing delay dominates).")
    print("- This is why embedded systems care about headroom and p99, not just p50.")


if __name__ == "__main__":
    main()
