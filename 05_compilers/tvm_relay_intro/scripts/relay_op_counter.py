from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter

DEFAULT_OPS = [
    "nn.conv2d",
    "nn.dense",
    "nn.batch_norm",
    "nn.relu",
    "nn.max_pool2d",
    "nn.avg_pool2d",
    "nn.global_avg_pool2d",
    "add(",
    "multiply(",
    "nn.softmax",
    "reshape(",
    "transpose(",
    "concatenate(",
]

def count_ops(text: str, ops: list[str]) -> Counter:
    c = Counter()
    for op in ops:
        c[op] = text.count(op)
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay_txt", type=str, required=True, help="Path to Relay IR text file")
    ap.add_argument("--ops", type=str, default="", help="Comma-separated extra ops to count")
    args = ap.parse_args()

    p = Path(args.relay_txt)
    text = p.read_text(encoding="utf-8", errors="ignore")

    ops = list(DEFAULT_OPS)
    if args.ops.strip():
        ops.extend([s.strip() for s in args.ops.split(",") if s.strip()])

    counts = count_ops(text, ops)

    print(f"Relay op count for: {p}")
    print("")
    for op, n in counts.most_common():
        if n > 0:
            print(f"{op:22s} {n}")

    total = sum(counts.values())
    print("")
    print(f"Total counted occurrences (subset): {total}")

if __name__ == "__main__":
    main()
