from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ops import make_candidate_ops

class MixedOp(nn.Module):
    """
    Mixed operation with architecture parameters alpha (logits).
    """
    def __init__(self, C_in: int, C_out: int, stride: int):
        super().__init__()
        self.ops = nn.ModuleDict(make_candidate_ops(C_in, C_out, stride))
        self.op_names = list(self.ops.keys())
        self.alpha = nn.Parameter(torch.zeros(len(self.op_names)))

    def forward(self, x):
        w = F.softmax(self.alpha, dim=0)
        out = 0.0
        for i, name in enumerate(self.op_names):
            out = out + w[i] * self.ops[name](x)
        return out

    def probs(self):
        return torch.softmax(self.alpha.detach(), dim=0)

    def best_op(self) -> str:
        i = int(torch.argmax(self.alpha.detach()).item())
        return self.op_names[i]
