from __future__ import annotations
import torch.nn as nn
from mixed_op import MixedOp

class Cell(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        # node1 takes input0
        self.op10 = MixedOp(C, C, stride=1)
        # node2 takes input0 and node1
        self.op20 = MixedOp(C, C, stride=1)
        self.op21 = MixedOp(C, C, stride=1)

    def forward(self, x):
        n1 = self.op10(x)
        n2 = self.op20(x) + self.op21(n1)
        return n2

    def mixed_ops(self):
        return [self.op10, self.op20, self.op21]
