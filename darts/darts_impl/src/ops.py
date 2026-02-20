from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out),
        )
    def forward(self, x): return self.op(x)

class Identity(nn.Module):
    def forward(self, x): return x

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.relu(x)
        y1 = self.conv1(x)
        y2 = self.conv2(x[:, :, 1:, 1:])
        return self.bn(torch.cat([y1, y2], dim=1))

def make_candidate_ops(C_in, C_out, stride):
    return {
        "skip": Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out),
        "conv3": ReLUConvBN(C_in, C_out, 3, stride=stride, padding=1),
        "conv5": ReLUConvBN(C_in, C_out, 5, stride=stride, padding=2),
    }
