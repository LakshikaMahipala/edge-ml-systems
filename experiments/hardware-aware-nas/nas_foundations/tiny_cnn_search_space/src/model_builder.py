from __future__ import annotations
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SqueezeExcite(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        mid = max(1, ch // r)
        self.fc1 = nn.Conv2d(ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, ch, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class MBConv(nn.Module):
    def __init__(self, cin, cout, k=3, exp=4, se=0, stride=1):
        super().__init__()
        mid = cin * exp
        self.use_res = (stride == 1 and cin == cout)
        self.pw1 = ConvBNAct(cin, mid, k=1, s=1)
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, k, stride=stride, padding=k//2, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcite(mid) if se else nn.Identity()
        self.pw2 = nn.Sequential(
            nn.Conv2d(mid, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
    def forward(self, x):
        y = self.pw1(x)
        y = self.dw(y)
        y = self.se(y)
        y = self.pw2(y)
        if self.use_res:
            y = y + x
        return F.relu(y, inplace=True)

class TinyNet(nn.Module):
    def __init__(self, arch: Dict[str, Any], input_size: int = 32):
        super().__init__()
        self.arch = arch
        stem = arch["stem_channels"]
        head = arch["head_channels"]
        num_classes = arch["num_classes"]

        self.stem = ConvBNAct(3, stem, k=3, s=1)

        layers: List[nn.Module] = []
        cin = stem
        # simple stage downsample at start of stage (stride=2) except stage0
        for si, st in enumerate(arch["stages"]):
            depth = int(st["depth"])
            cout = int(st["out_ch"])
            block = st["block"]
            k = int(st["k"])
            exp = int(st["exp"])
            se = int(st["se"])

            for di in range(depth):
                stride = 2 if (di == 0 and si > 0) else 1
                if block == "conv":
                    layers.append(ConvBNAct(cin, cout, k=k, s=stride))
                elif block == "mbconv":
                    layers.append(MBConv(cin, cout, k=k, exp=exp, se=se, stride=stride))
                else:
                    raise ValueError(f"unknown block: {block}")
                cin = cout

        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(cin, head, 1, bias=False),
            nn.BatchNorm2d(head),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(head, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

def build_model(arch: Dict[str, Any]) -> nn.Module:
    return TinyNet(arch)
