import argparse
import random
import numpy as np

def simulate(t_pre_ms, t_inf_ms, t_post_ms, arrival_rate_fps, n=2000, seed=0):
    random.seed(seed)

    # server availability times
    a_free = 0.0
    b_free = 0.0
    c_free = 0.0

    latencies = []

    t = 0.0
    for _ in range(n):
        # next arrival (Poisson)
        inter = random.expovariate(arrival_rate_fps)
        t += inter

        # Stage A
        a_start = max(t, a_free)
        a_end = a_start + (t_pre_ms / 1000.0)
        a_free = a_end

        # Stage B
        b_start = max(a_end, b_free)
        b_end = b_start + (t_inf_ms / 1000.0)
        b_free = b_end

        # Stage C
        c_start = max(b_end, c_free)
        c_end = c_start + (t_post_ms / 1000.0)
        c_free = c_end

        latencies.append((c_end - t) * 1000.0)  # ms

    arr = np.array(latencies, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_pre_ms", type=float, required=True)
    ap.add_argument("--t_inf_ms", type=float, required=True)
    ap.add_argument("--t_post_ms", type=float, required=True)
    ap.add_argument("--fps", type=float, required=True, help="arrival rate (frames per second)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = simulate(args.t_pre_ms, args.t_inf_ms, args.t_post_ms, args.fps, n=args.n, seed=args.seed)

    print("Pipeline Queue Simulation")
    print(f"Arrival fps: {args.fps}")
    print(f"Stage times (ms): pre={args.t_pre_ms}, inf={args.t_inf_ms}, post={args.t_post_ms}")
    print(out)

if __name__ == "__main__":
    main()
