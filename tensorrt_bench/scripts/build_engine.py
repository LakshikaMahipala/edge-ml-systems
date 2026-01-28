import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--engine", type=str, default="results/model_fp32.plan")
    ap.add_argument("--workspace_mb", type=int, default=1024)
    args = ap.parse_args()

    # We call trtexec (standard TensorRT CLI) because it is stable and widely used.
    # This avoids version-specific Python API pitfalls.
    cmd = (
        f"trtexec "
        f"--onnx={args.onnx} "
        f"--saveEngine={args.engine} "
        f"--workspace={args.workspace_mb} "
        f"--buildOnly "
    )

    print("RUN (later):")
    print(cmd)

    # Do not execute now; you will run later on the target machine.
    # If you later want auto-exec, we can add subprocess.run.

if __name__ == "__main__":
    main()
