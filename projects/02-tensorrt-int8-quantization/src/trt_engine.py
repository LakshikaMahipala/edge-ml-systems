"""Build and run TensorRT engines (FP32 / FP16 / INT8) from an ONNX model.

Uses the TensorRT 10 tensor-address API and PyTorch CUDA tensors for device
memory, so no PyCUDA is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)


class TorchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds calibration batches to TensorRT for INT8 post-training quantization.

    Entropy calibration picks, for each tensor, the clipping range that minimises
    the KL divergence between the FP32 activation histogram and its INT8 version.
    """

    def __init__(self, batches: Iterable[torch.Tensor], batch_size: int, cache_file: Path):
        super().__init__()
        self._batches = iter(batches)
        self._batch_size = batch_size
        self._cache_file = Path(cache_file)
        self._current: torch.Tensor | None = None  # keeps device memory alive

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names):  # noqa: ARG002 - signature fixed by TensorRT
        try:
            x = next(self._batches)
        except StopIteration:
            return None
        if x.shape[0] != self._batch_size:  # TensorRT needs full batches
            return None
        self._current = x.cuda().contiguous()
        return [int(self._current.data_ptr())]

    def read_calibration_cache(self):
        return self._cache_file.read_bytes() if self._cache_file.exists() else None

    def write_calibration_cache(self, cache) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache_file.write_bytes(bytes(cache))


def build_engine(onnx_path: Path, precision: str, max_batch: int, input_shape=(3, 224, 224),
                 calibrator: trt.IInt8Calibrator | None = None, workspace_gb: float = 2.0) -> bytes:
    """Returns a serialized engine. precision is 'fp32', 'fp16' or 'int8'."""
    builder = trt.Builder(LOGGER)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):  # needed on TensorRT 8
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, *input_shape), (max_batch, *input_shape), (max_batch, *input_shape))
    config.add_optimization_profile(profile)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if calibrator is None:
            raise ValueError("INT8 needs a calibrator")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # layers without INT8 kernels fall back to FP16
        config.int8_calibrator = calibrator
        config.set_calibration_profile(profile)
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed for {precision}")
    return bytes(serialized)


class TrtRunner:
    """Runs a single-input / single-output engine on PyTorch CUDA tensors."""

    def __init__(self, engine_bytes: bytes):
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.stream = torch.cuda.Stream()

    def _bind(self, x: torch.Tensor) -> torch.Tensor:
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        out = torch.empty(tuple(self.context.get_tensor_shape(self.output_name)),
                          dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        # Inputs/outputs were created on the default stream: order our stream after it.
        self.stream.wait_stream(torch.cuda.current_stream())
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.cuda().contiguous()
        out = self._bind(x)
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out

    def latency_ms(self, x: torch.Tensor, warmup: int = 20, iters: int = 200) -> list[float]:
        """Per-iteration GPU time of one inference (inputs already on the GPU)."""
        x = x.cuda().contiguous()
        self._bind(x)
        for _ in range(warmup):
            self.context.execute_async_v3(self.stream.cuda_stream)
        times = []
        for _ in range(iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(self.stream)
            self.context.execute_async_v3(self.stream.cuda_stream)
            end.record(self.stream)
            end.synchronize()
            times.append(start.elapsed_time(end))
        return times
