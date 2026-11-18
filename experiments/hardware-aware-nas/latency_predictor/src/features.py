from __future__ import annotations
from typing import Dict, Any
import math

def _get(d: Dict[str, Any], k: str, default=0):
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return float(default)

def graph_to_features(rec: Dict[str, Any]) -> Dict[str, float]:
    nodes = rec.get("nodes", [])
    globals_ = rec.get("globals", {})

    macs = 0.0
    params = 0.0
    bytes_moved = 0.0
    n_conv3 = n_conv5 = n_skip = n_other = 0.0
    max_c = 0.0
    mean_k = 0.0
    k_count = 0.0

    for n in nodes:
        op = n.get("op_type", "other")
        a = n.get("attrs", {})
        cin = _get(a, "cin", 0); cout = _get(a, "cout", 0)
        h = _get(a, "h", 0); w = _get(a, "w", 0)
        k = _get(a, "k", 1)

        if op.startswith("conv"):
            # conv MAC proxy: cout*h*w*cin*k*k
            macs += cout * h * w * cin * k * k
            params += cin * cout * k * k
            bytes_moved += (cin*h*w + cout*h*w)  # rough
            if int(k) == 3: n_conv3 += 1
            elif int(k) == 5: n_conv5 += 1
            else: n_other += 1
            max_c = max(max_c, cin, cout)
            mean_k += k
            k_count += 1
        elif op == "skip":
            n_skip += 1
        else:
            n_other += 1

    mean_k = (mean_k / max(k_count, 1.0))
    intensity = macs / max(bytes_moved, 1.0)

    # include global hints
    batch = _get(globals_, "batch", 1)
    # final feature dict
    return {
        "batch": batch,
        "depth_nodes": float(len(nodes)),
        "macs": float(macs),
        "params": float(params),
        "bytes": float(bytes_moved),
        "intensity": float(intensity),
        "n_conv3": float(n_conv3),
        "n_conv5": float(n_conv5),
        "n_skip": float(n_skip),
        "n_other": float(n_other),
        "max_channels": float(max_c),
        "mean_kernel": float(mean_k),
        "log_macs": math.log(macs + 1.0),
        "log_params": math.log(params + 1.0),
        "log_bytes": math.log(bytes_moved + 1.0),
    }

def features_to_vector(feat: Dict[str, float], keys: list[str]) -> list[float]:
    return [float(feat.get(k, 0.0)) for k in keys]
