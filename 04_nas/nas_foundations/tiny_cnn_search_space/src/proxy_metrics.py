from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def macs_proxy(model: nn.Module, input_shape=(1,3,32,32)) -> int:
    """
    Very rough MAC estimate by forward hooks.
    Counts MACs for Conv2d and Linear only.
    """
    macs = 0

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal macs
        x = inp[0]
        # out: (N, Cout, H, W)
        N, Cout, H, W = out.shape
        Cin = m.in_channels
        kH, kW = m.kernel_size
        groups = m.groups
        # per output element: (Cin/groups)*kH*kW multiplications
        macs += N * Cout * H * W * (Cin // groups) * kH * kW

    def lin_hook(m: nn.Linear, inp, out):
        nonlocal macs
        x = inp[0]
        N = x.shape[0]
        macs += N * m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))

    model.eval()
    x = torch.randn(*input_shape)
    _ = model(x)

    for h in hooks:
        h.remove()

    return int(macs)
