from __future__ import annotations
from typing import Dict, Any, List, Tuple

def arch_to_layers(arch: Dict[str, Any], input_hw: int = 32) -> List[Dict[str, Any]]:
    """
    Produce a simplified layer list with (type, Cin, Cout, H, W, k, groups).
    We treat stages as downsampling at start of stage (stride=2 for stage>0).
    """
    layers: List[Dict[str, Any]] = []
    H = W = input_hw
    Cin = 3

    # stem: conv3x3 to stem_channels
    stem = int(arch["stem_channels"])
    layers.append({"type":"conv", "cin":Cin, "cout":stem, "h":H, "w":W, "k":3, "groups":1})
    Cin = stem

    for si, st in enumerate(arch["stages"]):
        depth = int(st["depth"])
        Cout = int(st["out_ch"])
        block = st["block"]
        k = int(st["k"])
        exp = int(st["exp"])
        se = int(st["se"])

        for di in range(depth):
            stride2 = (di == 0 and si > 0)
            if stride2:
                H //= 2; W //= 2

            if block == "conv":
                layers.append({"type":"conv", "cin":Cin, "cout":Cout, "h":H, "w":W, "k":k, "groups":1})
                Cin = Cout
            elif block == "mbconv":
                mid = Cin * exp
                # pw expand
                layers.append({"type":"pw", "cin":Cin, "cout":mid, "h":H, "w":W, "k":1, "groups":1})
                # dw
                layers.append({"type":"dw", "cin":mid, "cout":mid, "h":H, "w":W, "k":k, "groups":mid})
                # se (ignored in ops, but we track as LUT-ish overhead)
                if se:
                    layers.append({"type":"se", "cin":mid, "cout":mid, "h":1, "w":1, "k":1, "groups":1})
                # pw project
                layers.append({"type":"pw", "cin":mid, "cout":Cout, "h":H, "w":W, "k":1, "groups":1})
                Cin = Cout
            else:
                raise ValueError(f"unknown block: {block}")

    # head 1x1
    head = int(arch["head_channels"])
    layers.append({"type":"pw", "cin":Cin, "cout":head, "h":H, "w":W, "k":1, "groups":1})
    Cin = head
    # classifier treated as FC with (Cin -> num_classes)
    layers.append({"type":"fc", "cin":Cin, "cout":int(arch["num_classes"]), "h":1, "w":1, "k":1, "groups":1})

    return layers

def layer_ops(layer: Dict[str, Any]) -> int:
    t = layer["type"]
    cin = int(layer["cin"]); cout = int(layer["cout"])
    h = int(layer["h"]); w = int(layer["w"])
    k = int(layer["k"]); groups = int(layer["groups"])

    if t in ("conv","pw","dw"):
        # MACs per output element: (cin/groups)*k*k
        return cout * h * w * (cin // groups) * k * k
    if t == "fc":
        return cin * cout
    if t == "se":
        # tiny MLP-ish overhead; keep small but nonzero
        return cin * 2
    return 0
