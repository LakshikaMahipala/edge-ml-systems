from __future__ import annotations
import torch
import torch.nn as nn

@torch.no_grad()
def compress_linear_svd(linear: nn.Linear, rank: int) -> nn.Sequential:
    """
    Replace a Linear(in->out) with Linear(in->rank) + Linear(rank->out)
    using SVD low-rank approximation.
    """
    W = linear.weight.data.clone()  # (out, in) in PyTorch
    b = linear.bias.data.clone() if linear.bias is not None else None

    # We want W^T shape (in, out) for our earlier math
    W_t = W.t()  # (in, out)

    # SVD
    U, S, Vh = torch.linalg.svd(W_t, full_matrices=False)
    U_r = U[:, :rank]         # (in, r)
    S_r = S[:rank]            # (r,)
    Vh_r = Vh[:rank, :]       # (r, out)

    # W_r = (U_r * S_r) @ Vh_r
    A = U_r * S_r             # (in, r)
    B = Vh_r                  # (r, out)

    # Build layers:
    # first: in -> r with weight A^T (r, in)
    # second: r -> out with weight B^T (out, r)
    l1 = nn.Linear(linear.in_features, rank, bias=False)
    l2 = nn.Linear(rank, linear.out_features, bias=(b is not None))

    l1.weight.data.copy_(A.t().contiguous())
    l2.weight.data.copy_(B.t().contiguous())
    if b is not None:
        l2.bias.data.copy_(b)

    return nn.Sequential(l1, l2)
