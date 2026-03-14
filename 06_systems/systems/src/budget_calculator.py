from __future__ import annotations
import argparse
import json
from pathlib import Path

from transfer_models import transfer_time_us
from example_configs import PCIE_GEN4_X16, ETH_100G_RDMA, ETH_10G_RPC, UART_1MBps

LINKS = {
    "pcie": PCIE_GEN4_X16,
    "rdma100g": ETH_100G_RDMA,
    "rpc10g": ETH_10G_RPC,
    "uart": UART_1MBps,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", type=str, default="pcie", choices=list(LINKS.keys()))
    ap.add_argument("--payload_kb", type=float, default=256.0)
    ap.add_argument("--compute_us", type=float, default=2000.0)
    ap.add_argument("--pre_us", type=float, default=200.0)
    ap.add_argument("--post_us", type=float, default=200.0)
    ap.add_argument("--out", type=str, default="results/budgets.json")
    args = ap.parse_args()

    payload_bytes = int(args.payload_kb * 1024)
    link = LINKS[args.link]

    h2d = transfer_time_us(payload_bytes, link)
    d2h = transfer_time_us(payload_bytes, link)

    total = args.pre_us + h2d + args.compute_us + d2h + args.post_us
    frac_transfer = (h2d + d2h) / total

    out = {
        "link": link.name,
        "payload_kb": args.payload_kb,
        "pre_us": args.pre_us,
        "compute_us": args.compute_us,
        "post_us": args.post_us,
        "h2d_us": h2d,
        "d2h_us": d2h,
        "total_us": total,
        "transfer_fraction": frac_transfer,
        "note": "proxy model: effective bandwidth + fixed overhead",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote:", args.out)
    print(out)

if __name__ == "__main__":
    main()
