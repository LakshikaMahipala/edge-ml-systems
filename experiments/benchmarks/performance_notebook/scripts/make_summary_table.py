import json
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="performance_notebook/results")
    args = ap.parse_args()

    p = Path(args.dir)
    files = sorted(p.glob("*.json"))
    if not files:
        print("No result JSON files found.")
        return

    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append(d)

    print("| backend | model | device | pre_p50 | inf_p50 | post_p50 | io_p50 | e2e_p50 |")
    print("|--------|-------|--------|--------:|--------:|---------:|------:|-------:|")
    for d in rows:
        p50 = d.get("p50_ms", {})
        def fmt(x): return "TBD" if x is None else f"{x:.3f}"
        print(
            f"| {d.get('backend')} | {d.get('model')} | {d.get('device')} | "
            f"{fmt(p50.get('pre'))} | {fmt(p50.get('inf'))} | {fmt(p50.get('post'))} | "
            f"{fmt(p50.get('io'))} | {fmt(p50.get('e2e'))} |"
        )

if __name__ == "__main__":
    main()
