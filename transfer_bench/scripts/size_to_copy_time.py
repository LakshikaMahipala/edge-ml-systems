import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size_mb", type=float, required=True)
    ap.add_argument("--bw_gbps", type=float, required=True)
    args = ap.parse_args()

    bytes_ = args.size_mb * 1024 * 1024
    bw_Bps = args.bw_gbps * 1e9
    t_s = bytes_ / bw_Bps
    t_ms = t_s * 1e3

    print("Copy time estimate")
    print(f"Size: {args.size_mb:.3f} MB")
    print(f"Bandwidth: {args.bw_gbps:.3f} GB/s")
    print(f"Estimated time: {t_ms:.3f} ms")

if __name__ == "__main__":
    main()
