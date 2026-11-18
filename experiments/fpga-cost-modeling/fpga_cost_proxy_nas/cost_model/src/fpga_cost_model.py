from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import math

from arch_to_layerseq import arch_to_layers, layer_ops

@dataclass
class FPGACost:
    cycles_proxy: float
    lut_proxy: float
    bram_proxy: float
    bw_proxy: float
    ops_total: float
    notes: str = "proxy only"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def estimate_fpga_cost(
    arch: Dict[str, Any],
    input_hw: int = 32,
    P: int = 64,
    act_buf_k: float = 0.002,
    weight_buf_k: float = 0.000002,
) -> FPGACost:
    """
    P = parallelism factor (bigger P => fewer cycles but more LUT pressure).
    act_buf_k, weight_buf_k scale BRAM proxy terms.
    """
    layers = arch_to_layers(arch, input_hw=input_hw)
    ops = 0.0
    lut = 0.0
    bram = 0.0
    bw = 0.0

    for L in layers:
        o = float(layer_ops(L))
        ops += o

        cin = float(L["cin"]); cout = float(L["cout"])
        h = float(L["h"]); w = float(L["w"])
        t = L["type"]

        # cycles proxy: ops / P
        # (we compute total later)

        # LUT proxy:
        # - grows with P (parallel datapaths)
        # - grows with channel widths
        lut += (0.01 * P) + (0.0005 * cin * cout)
        if t == "dw":
            lut += 0.002 * cin  # depthwise control/routing overhead
        if t == "se":
            lut += 0.05 * cin   # extra control logic proxy

        # BRAM proxy: activation buffering and weight storage pressure
        act_elems = cin * h * w
        w_elems = cin * cout * (L["k"] ** 2) if t in ("conv","pw") else (cin * (L["k"] ** 2) if t == "dw" else cin * cout)
        bram += act_buf_k * act_elems + weight_buf_k * w_elems

        # BW proxy: bytes moved, very rough
        bw += 0.000001 * (act_elems + w_elems)

    cycles = ops / float(P)

    # Normalize mildly (log scale) so rewards don’t explode
    cycles_p = math.log(cycles + 1.0)
    lut_p = math.log(lut + 1.0)
    bram_p = math.log(bram + 1.0)
    bw_p = math.log(bw + 1.0)

    return FPGACost(
        cycles_proxy=float(cycles_p),
        lut_proxy=float(lut_p),
        bram_proxy=float(bram_p),
        bw_proxy=float(bw_p),
        ops_total=float(ops),
        notes=f"P={P}, input={input_hw} (proxy)"
    )
