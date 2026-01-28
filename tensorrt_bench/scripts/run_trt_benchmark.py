import argparse
import subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=str, required=True)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--log", type=str, default="", help="If set, save trtexec output here")
    ap.add_argument("--run", action="store_true", help="Actually execute trtexec now (later when installed)")
    args = ap.parse_args()

    engine = Path(args.engine)

    cmd = [
        "trtexec",
        f"--loadEngine={str(engine)}",
        f"--warmUp={args.warmup}",
        f"--iterations={args.iters}",
        f"--shapes=input:{args.batch}x3x{args.input_size}x{args.input_size}",
        "--useSpinWait",
    ]

    print("CMD:")
    print(" ".join(cmd))

    if args.run:
        if args.log:
            log_path = Path(args.log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
            print(f"Saved log: {log_path}")
        else:
            subprocess.run(cmd, check=True)

    print("")
    print("Next:")
    print("- Parse the saved log for latency stats (p50/p99) using scripts/parse_trtexec_log.py")

if __name__ == "__main__":
    main()
