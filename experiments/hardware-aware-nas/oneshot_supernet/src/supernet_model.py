from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F
from supernet_blocks import SwitchableConv

class Supernet(nn.Module):
    def __init__(self, widths=(16, 24), kernels=(3, 5), num_classes: int = 10):
        super().__init__()
        C0 = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, C0, 3, padding=1, bias=False),
            nn.BatchNorm2d(C0),
            nn.ReLU(inplace=True),
        )
        self.b1 = SwitchableConv(C0, widths=widths, kernels=kernels)
        self.b2 = SwitchableConv(max(widths), widths=widths, kernels=kernels)
        self.b3 = SwitchableConv(max(widths), widths=widths, kernels=kernels)
        self.head = nn.Linear(max(widths), num_classes)

    def set_subnet(self, cfg):
        # cfg = {"b1":(k,w), "b2":(k,w), "b3":(k,w)}
        self.b1.set_active(*cfg["b1"])
        self.b2.set_active(*cfg["b2"])
        self.b3.set_active(*cfg["b3"])

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        # ensure channel compatibility: pad to max channels for next blocks
        x = pad_to_max(x, self.b2.C_in)
        x = self.b2(x)
        x = pad_to_max(x, self.b3.C_in)
        x = self.b3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        # slice/pad to max width for head
        x = pad_to_max_vec(x, self.head.in_features)
        return self.head(x)

def pad_to_max(x, C_max: int):
    import torch
    if x.shape[1] == C_max:
        return x
    pad = torch.zeros(x.shape[0], C_max - x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

def pad_to_max_vec(x, C_max: int):
    import torch
    if x.shape[1] == C_max:
        return x
    pad = torch.zeros(x.shape[0], C_max - x.shape[1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)
