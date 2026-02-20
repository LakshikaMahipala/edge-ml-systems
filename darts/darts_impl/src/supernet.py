from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F
from cell import Cell

class DARTSSupernet(nn.Module):
    def __init__(self, C: int = 16, num_cells: int = 4, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        self.cells = nn.ModuleList([Cell(C) for _ in range(num_cells)])
        self.head = nn.Linear(C, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for c in self.cells:
            x = c(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

    def arch_parameters(self):
        return [m.alpha for c in self.cells for m in c.mixed_ops()]

    def weight_parameters(self):
        arch_ids = set(id(p) for p in self.arch_parameters())
        return [p for p in self.parameters() if id(p) not in arch_ids]
