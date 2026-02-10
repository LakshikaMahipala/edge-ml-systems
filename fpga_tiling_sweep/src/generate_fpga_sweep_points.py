from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="examples/sweep_config.yaml")
    ap.add_argument("--out_jsonl", type=str, default="data/sweep_points.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    out_p = Path(args.out_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    baud = float(cfg["global"]["baud"])
    f_clk = float(cfg["global"]["f_clk_mhz"])
    II_values = cfg["global"]["II_values"]

    n = 0
    with out_p.open("w", encoding="utf-8") as f:
        # FC points
        for IN in cfg["fc"]["IN_values"]:
            for OUT in cfg["fc"]["OUT_values"]:
                for UNROLL in cfg["fc"]["UNROLL_values"]:
                    if UNROLL > IN:
                        continue
                    for II in II_values:
                        row = {
                            "op_type": "INT8_FC",
                            "IN": int(IN),
                            "OUT": int(OUT),
                            "UNROLL": int(UNROLL),
                            "II": int(II),
                            "baud": baud,
                            "f_clk_mhz": f_clk,
                        }
                        f.write(json.dumps(row) + "\n")
                        n += 1

        # DWConv1D points
        K = int(cfg["dwconv1d"]["K"])
        for C in cfg["dwconv1d"]["C_values"]:
            for L in cfg["dwconv1d"]["L_values"]:
                for UNROLL_C in cfg["dwconv1d"]["UNROLL_C_values"]:
                    if UNROLL_C > C:
                        continue
                    for II in II_values:
                        row = {
                            "op_type": "INT8_DWCONV1D_K3",
                            "C": int(C),
                            "L": int(L),
                            "K": K,
                            "UNROLL_C": int(UNROLL_C),
                            "II": int(II),
                            "baud": baud,
                            "f_clk_mhz": f_clk,
                        }
                        f.write(json.dumps(row) + "\n")
                        n += 1

    print(f"Wrote {n} sweep points to {out_p}")

if __name__ == "__main__":
    main()
