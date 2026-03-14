# inference_bench/src/cpp_preproc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

try:
    import cpp_preproc_ext  # built from cpp_preproc/python via pybind11
except Exception as e:
    cpp_preproc_ext = None  # type: ignore


@dataclass
class CppPreprocConfig:
    out_h: int = 224
    out_w: int = 224
    device: str = "cpu"


def available() -> bool:
    return cpp_preproc_ext is not None


def load_resize_normalize_to_torch(image_path: str, cfg: CppPreprocConfig) -> torch.Tensor:
    """
    Returns torch float32 tensor in NCHW.
    Note: currently performs a NumPy -> Torch conversion (one copy).
    Later we can optimize this to reduce copies.
    """
    if cpp_preproc_ext is None:
        raise RuntimeError(
            "cpp_preproc_ext not available. Build the C++ extension in cpp_preproc/ and ensure it is on PYTHONPATH."
        )

    arr = cpp_preproc_ext.load_resize_normalize(image_path, cfg.out_h, cfg.out_w)  # numpy float32 NCHW
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=np.float32)

    x = torch.from_numpy(arr).to(torch.device(cfg.device))
    return x
