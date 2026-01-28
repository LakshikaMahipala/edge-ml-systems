import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=str, required=True)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    # trtexec can report latency stats. We treat it as the ground truth baseline tool.
    cmd = (
        f"trtexec "
        f"--loadEngine={args.engine} "
        f"--warmUp={args.warmup} "
        f"--iterations={args.iters} "
        f"--shapes=input:{args.batch}x3x224x224 "
        f"--useSpinWait "
        f"--noDataTransfers=0 "
    )

    print("RUN (later):")
    print(cmd)
    print("")
    print("Record from output:")
    print("- Throughput (qps / inf/s)")
    print("- Latency (mean / p50 / p99 if shown)")
    print("- GPU compute time vs H2D/D2H if reported")

if __name__ == "__main__":
    main()
