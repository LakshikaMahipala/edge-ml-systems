from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class Hooks:
    before_batch: Callable[[int], None]
    after_batch: Callable[[int, torch.Tensor], None]


def default_hooks() -> Hooks:
    def _before(_i: int) -> None:
        return

    def _after(_i: int, _logits: torch.Tensor) -> None:
        return

    return Hooks(before_batch=_before, after_batch=_after)


def resolve_device(device: str) -> torch.device:
    d = device.lower().strip()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda")
    if d == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    raise ValueError("device must be auto|cpu|cuda")


def build_model(name: str, pretrained: bool) -> nn.Module:
    name = name.lower().strip()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("model must be resnet18")
    return m.eval()
