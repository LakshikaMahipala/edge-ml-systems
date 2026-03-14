from __future__ import annotations
import torch.nn as nn

def conv_bn_relu(C_in, C_out, k, stride=1):
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(C_in, C_out, k, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
    )

class Identity(nn.Module):
    def forward(self, x): return x
