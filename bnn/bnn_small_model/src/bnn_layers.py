from __future__ import annotations
import torch
import torch.nn as nn

class STE_Sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.sign().clamp(min=-1.0, max=1.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        # straight-through estimator: pass gradient only where |x| <= 1
        mask = (x.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * mask

def ste_sign(x: torch.Tensor) -> torch.Tensor:
    return STE_Sign.apply(x)

class BinaryLinear(nn.Module):
    """
    Binary weights and binary activations (in forward).
    Backprop uses STE through sign.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_b = ste_sign(x)
        w_b = ste_sign(self.weight)
        out = x_b @ w_b.t()
        if self.bias is not None:
            out = out + self.bias
        return out
