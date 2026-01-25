# inference_bench/src/preproc_paths.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from inference_bench.src.pytorch_infer import PyTorchRunner
from inference_bench.src.cpp_preproc import CppPreprocConfig, available as cpp_available, load_resize_normalize_to_torch


@dataclass
class PreprocPathConfig:
    input_size: int = 224
    device: str = "cpu"


def python_preprocess_fn(runner: PyTorchRunner) -> Callable[[], torch.Tensor]:
    """
    Returns a callable that produces a preprocessed tensor using the runner's Python preprocess.
    Important: runner.preprocess() should return NCHW float32 on the target device.
    """
    def _fn() -> torch.Tensor:
        return runner.preprocess()
    return _fn


def cpp_preprocess_fn(image_path: str, cfg: PreprocPathConfig) -> Callable[[], torch.Tensor]:
    """
    Returns a callable that produces a preprocessed tensor using the C++ extension.
    """
    if not cpp_available():
        raise RuntimeError("cpp_preproc_ext not available. Build the C++ extension and ensure it is importable.")

    c_cfg = CppPreprocConfig(out_h=cfg.input_size, out_w=cfg.input_size, device=cfg.device)

    def _fn() -> torch.Tensor:
        return load_resize_normalize_to_torch(image_path, c_cfg)
    return _fn
