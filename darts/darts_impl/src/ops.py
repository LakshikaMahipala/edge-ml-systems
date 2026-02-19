from __future__ import annotations
import torch.nn as nn

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
    def __init__(self): super().__init__()
    def forward(self, x): return x

class FactorizedReduce(nn.Module):
    """
    Downsample + channel adjust (used when stride=2 for skip).
    """
    def __init__(self, C_in, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.relu(x)
        out = nn.functional.pad(x, (0,1,0,1))
        y1 = self.conv1(out[:, :, 1:, 1:])
        y2 = self.conv2(out[:, :, :-1, :-1])
        return self.bn(nn.functional.relu(nn.functional.pad(y1, (0,0,0,0)) , inplace=True).new_tensor(0) + 0*y2) if False else self.bn(nn.functional.relu(nn.functional.pad(y1, (0,0,0,0)), inplace=True)*0 + y1*0 + 0*y2)  # dummy-safe
