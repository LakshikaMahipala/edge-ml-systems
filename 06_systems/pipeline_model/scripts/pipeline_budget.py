import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_pre_ms", type=float, required=True)
    ap.add_argument("--t_inf_ms", type=float, required=True)
    ap.add_argument("--t_post_ms", type=float, required=True)
    ap.add_argument("--copies_ms", type=float, default=0.0, help="optional copy overhead")
    args = ap.parse_args()

    t_pre = args.t_pre_ms
    t_inf = args.t_inf_ms
    t_post = args.t_post_ms
    t_copy = args.copies_ms

    e2e_latency = t_pre + t_inf + t_post + t_copy

    # sequential TPS
    tps_seq = 1000.0 / e2e_latency if e2e_latency > 0 else float("inf")

    # pipelined TPS (one worker per stage)
    bottleneck = max(t_pre, t_inf, t_post)
    tps_pipe = 1000.0 / bottleneck if bottleneck > 0 else float("inf")

    print("Pipeline Budget")
    print(f"Stage times (ms): pre={t_pre:.3f}, inf={t_inf:.3f}, post={t_post:.3f}, copy={t_copy:.3f}")
    print(f"E2E latency estimate (ms): {e2e_latency:.3f}")
    print(f"Sequential throughput (items/s): {tps_seq:.2f}")
    print(f"Pipelined throughput (items/s):  {tps_pipe:.2f}")
    print(f"Bottleneck stage time (ms): {bottleneck:.3f}")

if __name__ == "__main__":
    main()
