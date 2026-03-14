import argparse
import json
from pathlib import Path

def speedup(p: float, S: float) -> float:
    return 1.0 / ((1.0 - p) + (p / S))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_pre_ms", type=float, required=True)
    ap.add_argument("--t_inf_ms", type=float, required=True)
    ap.add_argument("--t_post_ms", type=float, required=True)
    ap.add_argument("--t_copy_ms", type=float, default=0.0)
    ap.add_argument("--S_list", type=str, default="1,1.5,2,3,5,10,1000")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    t_total = args.t_pre_ms + args.t_inf_ms + args.t_post_ms + args.t_copy_ms
    if t_total <= 0:
        raise ValueError("Total time must be > 0")

    p = args.t_inf_ms / t_total
    serial_frac = 1.0 - p
    max_speedup = 1.0 / serial_frac if serial_frac > 0 else float("inf")

    S_values = [float(x.strip()) for x in args.S_list.split(",") if x.strip()]
    rows = []
    for S in S_values:
        rows.append({
            "S": S,
            "speedup": speedup(p, S),
            "new_total_ms": t_total / speedup(p, S)
        })

    report = {
        "inputs_ms": {
            "pre": args.t_pre_ms,
            "inf": args.t_inf_ms,
            "post": args.t_post_ms,
            "copy": args.t_copy_ms,
            "total": t_total
        },
        "amdahl": {
            "p_accel_fraction": p,
            "serial_fraction": serial_frac,
            "max_speedup_inf_to_infinity": max_speedup
        },
        "speedup_table": rows
    }

    print(json.dumps(report, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
