import json
from pathlib import Path
import argparse
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet18")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    # This is a stub schema. Later we will load real PerfSummary JSON.
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.backend,
        "model": args.model,
        "device": args.device,
        "p50_ms": {"pre": None, "inf": None, "post": None, "io": None, "e2e": None},
        "p99_ms": {"pre": None, "inf": None, "post": None, "io": None, "e2e": None},
        "notes": "Fill with real numbers after running locally.",
    }

    text = json.dumps(record, indent=2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved: {out_path}")
    else:
        print(text)

if __name__ == "__main__":
    main()
