from __future__ import annotations
from typing import Dict, Any, List
from supernet import DARTSSupernet

def extract_genotype(net: DARTSSupernet) -> Dict[str, Any]:
    genes: List[Dict[str, Any]] = []
    edge_idx = 0
    for ci, cell in enumerate(net.cells):
        for mi, mop in enumerate(cell.mixed_ops()):
            genes.append({
                "cell": ci,
                "edge": edge_idx,
                "best_op": mop.best_op(),
                "probs": [float(x) for x in mop.probs().tolist()],
                "op_names": mop.op_names,
            })
            edge_idx += 1
    return {"genotype": genes}
