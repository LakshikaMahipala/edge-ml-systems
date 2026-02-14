from __future__ import annotations
from typing import Dict, Any, Tuple
import math
import torch

# run later with PYTHONPATH pointing to nas_foundations/tiny_cnn_search_space/src
from model_builder import build_model
from proxy_metrics import count_params, macs_proxy

def to_arch_format(sampled: Dict[str, Any]) -> Dict[str, Any]:
    # sampled contains discrete choices like stem_channels, head_channels, stage{i}_depth, etc.
    # build the stage list
    stages = []
    for i in range(3):
        stages.append({
            "depth": int(sampled[f"stage{i}_depth"]),
            "out_ch": int(sampled[f"stage{i}_out_ch"]),
            "block": sampled[f"stage{i}_block"],
            "k": int(sampled[f"stage{i}_k"]),
            "exp": int(sampled[f"stage{i}_exp"]),
            "se": int(sampled[f"stage{i}_se"]),
        })
    return {
        "stem_channels": int(sampled["stem_channels"]),
        "head_channels": int(sampled["head_channels"]),
        "num_classes": int(sampled["num_classes"]),
        "stages": stages,
    }

def acc_proxy(params: int, macs: int) -> float:
    """
    Synthetic 'accuracy proxy' just to let RL training run.
    Later replace with real validation accuracy.
    We assume accuracy increases with model capacity but saturates.
    """
    # normalize
    p = math.log(params + 1.0)
    m = math.log(macs + 1.0)
    # saturating function
    return float(1.0 - math.exp(-0.08 * (0.6 * p + 0.4 * m)))

def reward_from_arch(arch: Dict[str, Any], lam_p=0.15, lam_m=0.15) -> Tuple[float, Dict[str, float]]:
    model = build_model(arch)
    params = count_params(model)
    macs = macs_proxy(model)

    a = acc_proxy(params, macs)

    # normalize penalties (log scale)
    p_term = math.log(params + 1.0) / 20.0
    m_term = math.log(macs + 1.0) / 25.0

    r = a - lam_p * p_term - lam_m * m_term

    comps = {
        "acc_proxy": float(a),
        "params": float(params),
        "macs": float(macs),
        "p_term": float(p_term),
        "m_term": float(m_term),
        "reward": float(r),
    }
    return float(r), comps
