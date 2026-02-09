from __future__ import annotations
import argparse
from pathlib import Path

def toy_rewrite(text: str) -> str:
    # Extremely naive examples for learning:
    # 1) rename "nn.relu(" to "nn.relu_fused(" (not a real op)
    # 2) rename "add(" to "add_fused(" when near relu
    # This is just to visualize the concept of a rewrite pass.

    lines = text.splitlines()
    out = []
    prev_was_relu = False

    for ln in lines:
        s = ln
        if "nn.relu(" in s:
            prev_was_relu = True
            s = s.replace("nn.relu(", "nn.relu_fused(")
        else:
            if prev_was_relu and "add(" in s:
                s = s.replace("add(", "add_fused(")
            prev_was_relu = False
        out.append(s)

    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", type=str, required=True)
    ap.add_argument("--out_txt", type=str, required=True)
    args = ap.parse_args()

    in_p = Path(args.in_txt)
    out_p = Path(args.out_txt)

    text = in_p.read_text(encoding="utf-8", errors="ignore")
    new_text = toy_rewrite(text)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(new_text, encoding="utf-8")
    print(f"Wrote rewritten (toy) Relay text to: {out_p}")

if __name__ == "__main__":
    main()
