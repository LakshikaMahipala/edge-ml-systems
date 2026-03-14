import argparse
from pathlib import Path
import numpy as np

# TensorRT imports (will work later when TRT is installed)
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class NpyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Feeds calibration batches from an NPY file of shape (N,3,H,W), float32.
    """
    def __init__(self, npy_path: str, input_name: str, batch_size: int, cache_file: str):
        super().__init__()
        self.npy_path = npy_path
        self.input_name = input_name
        self.batch_size = batch_size
        self.cache_file = cache_file

        self.data = np.load(npy_path)  # (N,3,H,W) float32
        self.idx = 0

        # Allocate device memory for one batch
        self.device_input = cuda.mem_alloc(self.data[0:batch_size].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > len(self.data):
            return None

        batch = self.data[self.idx:self.idx + self.batch_size]
        self.idx += self.batch_size

        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        p = Path(self.cache_file)
        if p.exists():
            return p.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        Path(self.cache_file).write_bytes(cache)


def build_int8_engine(onnx_path: str, engine_path: str, calib_npy: str, calib_cache: str,
                      input_name: str, batch_size: int, workspace_mb: int):
    onnx_path = str(onnx_path)
    engine_path = str(engine_path)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        onnx_bytes = Path(onnx_path).read_bytes()
        if not parser.parse(onnx_bytes):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

        # Enable INT8
        config.set_flag(trt.BuilderFlag.INT8)

        # Calibrator (implicit quantization via calibration cache)
        calibrator = NpyCalibrator(
            npy_path=calib_npy,
            input_name=input_name,
            batch_size=batch_size,
            cache_file=calib_cache
        )
        config.int8_calibrator = calibrator

        # Build engine
        print("Building INT8 engine (this will run calibration the first time)...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build INT8 engine")

        Path(engine_path).write_bytes(serialized_engine)
        print(f"Saved INT8 engine: {engine_path}")
        print(f"Calibration cache (written/updated): {calib_cache}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--engine", type=str, default="results/model_int8.plan")
    ap.add_argument("--calib_npy", type=str, default="results/calib/calib_fp32.npy")
    ap.add_argument("--calib_cache", type=str, default="results/calib/calib.cache")
    ap.add_argument("--input_name", type=str, default="input")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workspace_mb", type=int, default=2048)
    args = ap.parse_args()

    Path(args.engine).parent.mkdir(parents=True, exist_ok=True)
    Path(args.calib_cache).parent.mkdir(parents=True, exist_ok=True)

    build_int8_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        calib_npy=args.calib_npy,
        calib_cache=args.calib_cache,
        input_name=args.input_name,
        batch_size=args.batch,
        workspace_mb=args.workspace_mb,
    )


if __name__ == "__main__":
    main()
