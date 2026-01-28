import argparse
import subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--engine", type=str, default="results/model_fp32.plan")
    ap.add_argument("--workspace_mb", type=int, default=1024)
    ap.add_argument("--fp16", action="store_true", help="Enable FP16 build")
    ap.add_argument("--verbose", action="store_true", help="Verbose TensorRT build log")
    ap.add_argument("--run", action="store_true", help="Actually execute trtexec now (later when TensorRT is installed)")
    args = ap.parse_args()

    onnx = Path(args.onnx)
    engine = Path(args.engine)
    engine.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "trtexec",
        f"--onnx={str(onnx)}",
        f"--saveEngine={str(engine)}",
        f"--workspace={args.workspace_mb}",
        "--buildOnly",
    ]

    if args.fp16:
        cmd.append("--fp16")

    if args.verbose:
        cmd.append("--verbose")

    print("CMD:")
    print(" ".join(cmd))

    if args.run:
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
