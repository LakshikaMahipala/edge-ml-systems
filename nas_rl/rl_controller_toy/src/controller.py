from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.distributions import Categorical

@dataclass
class Sampled:
    arch: Dict[str, Any]
    logprob: torch.Tensor

class NASController(nn.Module):
    """
    Toy controller: independent categorical distributions for each choice.
    This is enough to demonstrate REINFORCE.
    """
    def __init__(self, space_spec: Dict[str, List[Any]]):
        super().__init__()
        self.space_spec = space_spec
        self.keys = list(space_spec.keys())
        self.logits = nn.ParameterDict()
        for k in self.keys:
            n = len(space_spec[k])
            self.logits[k] = nn.Parameter(torch.zeros(n))

    def sample(self) -> Sampled:
        arch = {}
        logps = []
        for k in self.keys:
            dist = Categorical(logits=self.logits[k])
            idx = dist.sample()
            logps.append(dist.log_prob(idx))
            arch[k] = self.space_spec[k][int(idx)]
        return Sampled(arch=arch, logprob=torch.stack(logps).sum())

    def greedy(self) -> Dict[str, Any]:
        arch = {}
        for k in self.keys:
            idx = int(torch.argmax(self.logits[k]).item())
            arch[k] = self.space_spec[k][idx]
        return arch
