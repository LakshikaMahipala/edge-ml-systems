from __future__ import annotations
from typing import Dict, Any, Tuple
import torch

from controller import NASController
from reward_proxy import to_arch_format, reward_from_arch

def do_rollout(ctrl: NASController, lam_p=0.15, lam_m=0.15) -> Tuple[Dict[str, Any], torch.Tensor, float, Dict[str, float]]:
    s = ctrl.sample()
    arch = to_arch_format(s.arch)
    r, comps = reward_from_arch(arch, lam_p=lam_p, lam_m=lam_m)
    return arch, s.logprob, r, comps
