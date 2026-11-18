from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchableConv(nn.Module):
    """
    A single block that supports:
    - kernel size choice: 3 or 5
    - output channel choice: among widths
    We implement this by allocating MAX resources and slicing.
    """
    def __init__(self, C_in: int, widths=(16, 24), kernels=(3, 5)):
        super().__init__()
        self.C_in = C_in
        self.widths = list(widths)
        self.kernels = list(kernels)

        self.C_out_max = max(self.widths)
        self.k_max = max(self.kernels)

        # One max conv; we use padding and slice weights logically via masking idea.
        # For simplicity: use separate convs per kernel but share output slicing.
        self.conv3 = nn.Conv2d(C_in, self.C_out_max, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(C_in, self.C_out_max, 5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(self.C_out_max)

        self.active_kernel = 3
        self.active_width = self.C_out_max

    def set_active(self, kernel: int, width: int):
        assert kernel in self.kernels
        assert width in self.widths
        self.active_kernel = kernel
        self.active_width = width

    def forward(self, x):
        if self.active_kernel == 3:
            y = self.conv3(x)
        else:
            y = self.conv5(x)
        y = self.bn(y)
        y = F.relu(y, inplace=True)
        # slice channels for smaller width
        return y[:, : self.active_width, :, :]
