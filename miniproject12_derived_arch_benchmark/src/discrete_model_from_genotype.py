from __future__ import annotations
import json
import torch.nn as nn
import torch.nn.functional as F
from discrete_cell_from_genotype import DiscreteCell

class DiscreteDARTSModel(nn.Module):
    def __init__(self, C: int = 16, num_cells: int = 4, num_classes: int = 10, edge_ops=None):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.cells = nn.ModuleList([DiscreteCell(C, edge_ops) for _ in range(num_cells)])
        self.head = nn.Linear(C, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for c in self.cells:
            x = c(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

def edge_ops_from_genotype_json(path: str) -> list[str]:
    """
    genotype is list of edges with best_op.
    We take the first 3 edges (our tiny cell definition).
    """
    obj = json.loads(open(path, "r", encoding="utf-8").read())
    genes = obj["genotype"] if "genotype" in obj else obj["geno"]["genotype"]
    ops = [g["best_op"] for g in genes[:3]]
    return ops
