from __future__ import annotations
import torch.nn as nn
from discrete_ops import conv_bn_relu, Identity

def op_from_name(name: str, C: int):
    if name == "skip":
        return Identity()
    if name == "conv3":
        return conv_bn_relu(C, C, 3)
    if name == "conv5":
        return conv_bn_relu(C, C, 5)
    raise ValueError(f"unknown op: {name}")

class DiscreteCell(nn.Module):
    def __init__(self, C: int, edge_ops: list[str]):
        """
        edge_ops length must be 3 for our tiny cell edges:
        [op10, op20, op21]
        """
        super().__init__()
        assert len(edge_ops) == 3
        self.op10 = op_from_name(edge_ops[0], C)
        self.op20 = op_from_name(edge_ops[1], C)
        self.op21 = op_from_name(edge_ops[2], C)

    def forward(self, x):
        n1 = self.op10(x)
        n2 = self.op20(x) + self.op21(n1)
        return n2
