import argparse
from pathlib import Path
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def load_engine(engine_path: str) -> trt.ICudaEngine:
    runtime = trt.Runtime(TRT_LOGGER)
    engine_bytes = Path(engine_path).read_bytes()
    return runtime.deserialize_cuda_engine(engine_bytes)


def infer(engine: trt.ICudaEngine, x: np.ndarray, input_name="input") -> np.ndarray:
    """
    x: (B,3,H,W) float32
    returns: logits (B,1000) float32 (if model is ImageNet classifier)
    """
    assert x.dtype == np.float32
    context = engine.create_execution_context()

    # Resolve bindings
    input_idx = engine.get_binding_index(input_name)
    output_idx = 1 - input_idx  # assumes only 1 input + 1 output

    # Shapes
    context.set_binding_shape(input_idx, x.shape)

    out_shape = tuple(context.get_binding_shape(output_idx))
    out = np.empty(out_shape, dtype=np.float32)

    # Allocate device buffers
    d_in = cuda.mem_alloc(x.nbytes)
    d_out = cuda.mem_alloc(out.nbytes)

    cuda.memcpy_htod(d_in, x)

    bindings = [0] * engine.num_bindings
    bindings[input_idx] = int(d_in)
    bindings[output_idx] = int(d_out)

    context.execute_v2(bindings)
    cuda.memcpy_dtoh(out, d_out)
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # per-sample cosine similarity
    a2 = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b2 = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(a2 * b2, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32_engine", type=str, required=True)
    ap.add_argument("--int8_engine", type=str, required=True)
    ap.add_argument("--calib_npy", type=str, default="results/calib/calib_fp32.npy")
    ap.add_argument("--n", type=int, default=128, help="how many samples to compare")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--input_name", type=str, default="input")
    args = ap.parse_args()

    X = np.load(args.calib_npy).astype(np.float32)
    X = X[:args.n]

    fp32 = load_engine(args.fp32_engine)
    int8 = load_engine(args.int8_engine)

    top1_match = 0
    cos_sims = []

    for i in range(0, len(X), args.batch):
        xb = X[i:i+args.batch]
        y_fp32 = infer(fp32, xb, input_name=args.input_name)
        y_int8 = infer(int8, xb, input_name=args.input_name)

        p_fp32 = np.argmax(y_fp32, axis=1)
        p_int8 = np.argmax(y_int8, axis=1)

        top1_match += int(np.sum(p_fp32 == p_int8))
        cos_sims.extend(cosine_sim(y_fp32, y_int8).tolist())

    acc_proxy = top1_match / len(X)
    cos_mean = float(np.mean(cos_sims))

    print("INT8 vs FP32 agreement (proxy)")
    print(f"Samples: {len(X)}")
    print(f"Top-1 agreement: {acc_proxy:.4f}")
    print(f"Mean cosine(logits): {cos_mean:.4f}")
    print("")
    print("Note: true accuracy delta requires the correct labeled dataset (e.g., ImageNet val).")

if __name__ == "__main__":
    main()
