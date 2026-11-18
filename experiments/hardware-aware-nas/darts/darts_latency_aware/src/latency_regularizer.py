from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple, Dict

from latency_table import LatencyTable

def expected_latency_from_mixedop(alpha: torch.Tensor, op_names, table: LatencyTable) -> torch.Tensor:
    p = F.softmax(alpha, dim=0)
    costs = torch.tensor([table.op_latency[n] for n in op_names], dtype=p.dtype, device=p.device)
    return torch.sum(p * costs)

def supernet_expected_latency(net, table: LatencyTable) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Sum expected latency over all MixedOps in all cells.
    """
    total = torch.tensor(0.0)
    comps = {}
    idx = 0
    for cell in net.cells:
        for mop in cell.mixed_ops():
            e = expected_latency_from_mixedop(mop.alpha, mop.op_names, table)
            total = total + e
            comps[f"edge_{idx}"] = float(e.detach().item())
            idx += 1
    comps["total"] = float(total.detach().item())
    return total, comps
