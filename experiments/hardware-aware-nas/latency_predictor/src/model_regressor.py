from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPRegressor(nn.Module):
    def __init__(self, d_in: int, d_h: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_h)
        self.out = nn.Linear(d_h, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(1)
