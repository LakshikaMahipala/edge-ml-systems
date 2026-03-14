from __future__ import annotations
import json
import torch.nn as nn
import torch.nn.functional as F
from discrete_ops import conv_bn_relu

class DiscreteOneShotModel(nn.Module):
    def __init__(self, cfg, num_classes: int = 10):
        super().__init__()
        # cfg keys: b1,b2,b3 each = [k,width]
        C0 = 16
        self.stem = conv_bn_relu(3, C0, 3)
        k1, w1 = cfg["b1"]
        k2, w2 = cfg["b2"]
        k3, w3 = cfg["b3"]
        self.b1 = conv_bn_relu(C0, w1, k1)
        self.b2 = conv_bn_relu(w1, w2, k2)
        self.b3 = conv_bn_relu(w2, w3, k3)
        self.head = nn.Linear(w3, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

def load_subnet_cfg(path: str):
    obj = json.loads(open(path, "r", encoding="utf-8").read())
    if "subnet" in obj:
        return obj["subnet"]
    if "cfg" in obj:
        return obj["cfg"]
    return obj
