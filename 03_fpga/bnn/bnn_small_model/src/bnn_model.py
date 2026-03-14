from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from bnn_layers import BinaryLinear

class BNNMLP(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 128, num_classes: int = 4):
        super().__init__()
        self.fc1 = BinaryLinear(in_dim, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)

        self.fc2 = BinaryLinear(hidden, hidden, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden)

        # Often keep last layer real-valued for accuracy.
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.fc1(x))
        x = F.hardtanh(x)   # keep activations bounded (helps binarization)
        x = self.bn2(self.fc2(x))
        x = F.hardtanh(x)
        return self.fc3(x)
