# inference_bench/src/pytorch_infer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class PyTorchConfig:
    model_name: str = "resnet18"
    device: str = "cpu"
    input_size: int = 224
    num_classes: int = 1000
    topk: int = 5
    channels: int = 3
    batch: int = 1


class PyTorchRunner:
    """
    Minimal PyTorch inference pipeline:
    preprocess (create tensor + normalize) -> forward -> postprocess (topk)
    We keep it deterministic and self-contained (no dataset needed yet).
    """

    def __init__(self, cfg: PyTorchConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.model = self._load_model(cfg.model_name).to(self.device).eval()

        # Standard ImageNet normalization (common baseline)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        # Pre-allocate an input tensor to reduce allocation noise
        self._x = torch.rand(
            (cfg.batch, cfg.channels, cfg.input_size, cfg.input_size),
            dtype=torch.float32,
            device=self.device,
        )

    
    def _load_model(self, name: str) -> torch.nn.Module:
        import torchvision.models as models

        if name == "resnet18":
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if name == "mobilenet_v3_small":
            return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        if name == "efficientnet_b0":
            return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        raise ValueError(f"Unsupported model_name: {name}")

    @torch.inference_mode()
    def preprocess(self) -> torch.Tensor:
        # Use the pre-allocated tensor, normalize in-place style (creates new tensor ops internally)
        x = self._x
        x = (x - self.mean) / self.std
        return x

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits

    @torch.inference_mode()
    def postprocess(self, logits: torch.Tensor) -> Tuple[List[int], List[float]]:
        probs = F.softmax(logits, dim=1)
        topk = min(self.cfg.topk, probs.shape[1])
        vals, idx = torch.topk(probs, k=topk, dim=1)
        return idx[0].tolist(), vals[0].tolist()

    @torch.inference_mode()
    def end_to_end(self) -> Tuple[List[int], List[float]]:
        x = self.preprocess()
        logits = self.forward(x)
        return self.postprocess(logits)
