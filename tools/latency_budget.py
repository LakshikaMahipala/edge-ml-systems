import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_pre_ms", type=float, required=True)
    ap.add_argument("--t_post_ms", type=float, required=True)
    ap.add_argument("--t_copy_in_ms", type=float, required=True)
    ap.add_argument("--t_copy_out_ms", type=float, required=True)
    ap.add_argument("--t_compute_cpu_ms", type=float, required=True, help="baseline CPU inference time")
    ap.add_argument("--t_compute_accel_ms", type=float, required=True, help="accelerator compute time")
    args = ap.parse_args()

    t_cpu_total = args.t_pre_ms + args.t_compute_cpu_ms + args.t_post_ms
    t_accel_total = args.t_pre_ms + args.t_copy_in_ms + args.t_compute_accel_ms + args.t_copy_out_ms + args.t_post_ms

    speedup = t_cpu_total / t_accel_total if t_accel_total > 0 else float("inf")

    print("Latency Budget")
    print(f"CPU total (ms):   {t_cpu_total:.3f}")
    print(f"ACCEL total (ms): {t_accel_total:.3f}")
    print(f"Speedup:          {speedup:.3f}x")
    print("")
    print("Breakdown (ACCEL path):")
    print(f"  pre     : {args.t_pre_ms:.3f} ms")
    print(f"  copy_in : {args.t_copy_in_ms:.3f} ms")
    print(f"  compute : {args.t_compute_accel_ms:.3f} ms")
    print(f"  copy_out: {args.t_copy_out_ms:.3f} ms")
    print(f"  post    : {args.t_post_ms:.3f} ms")

if __name__ == "__main__":
    main()
