from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F
from discrete_ops import conv_bn_relu

class BaselineSmallCNN(nn.Module):
    def __init__(self, C: int = 16, num_classes: int = 10):
        super().__init__()
        self.f1 = conv_bn_relu(3, C, 3)
        self.f2 = conv_bn_relu(C, 2*C, 3)
        self.f3 = conv_bn_relu(2*C, 2*C, 3)
        self.head = nn.Linear(2*C, num_classes)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)
