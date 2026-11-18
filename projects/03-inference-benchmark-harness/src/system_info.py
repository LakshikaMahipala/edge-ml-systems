# inference_bench/src/system_info.py
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import torch
except Exception:
    torch = None  # type: ignore


@dataclass
class SystemInfo:
    python: str
    platform: str
    processor: str
    cpu_count: int
    torch: str
    cuda_available: bool
    cuda_version: str
    device_name: str

    def to_dict(self) -> dict:
        return {
            "python": self.python,
            "platform": self.platform,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
            "torch": self.torch,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "device_name": self.device_name,
        }


def get_system_info(device: str = "cpu") -> SystemInfo:
    py = sys.version.split()[0]
    plat = platform.platform()
    proc = platform.processor() or "unknown"
    cpu_count = os.cpu_count() or 0

    torch_ver = "not_installed"
    cuda_avail = False
    cuda_ver = "n/a"
    dev_name = device

    if torch is not None:
        torch_ver = torch.__version__
        cuda_avail = bool(torch.cuda.is_available())
        cuda_ver = getattr(torch.version, "cuda", None) or "n/a"

        if device.startswith("cuda") and cuda_avail:
            try:
                dev_name = torch.cuda.get_device_name(0)
            except Exception:
                dev_name = "cuda"

    return SystemInfo(
        python=py,
        platform=plat,
        processor=proc,
        cpu_count=cpu_count,
        torch=torch_ver,
        cuda_available=cuda_avail,
        cuda_version=cuda_ver,
        device_name=dev_name,
    )
