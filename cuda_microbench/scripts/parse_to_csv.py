import re
import sys
from pathlib import Path

# Usage:
# python scripts/parse_to_csv.py results/run_YYYYMMDD_HHMMSS.txt results/summary.csv

PAT_BENCH = re.compile(r"^==\s*(.+?)\s*==\s*$")
PAT_TIME  = re.compile(r"(Kernel avg time|GEMM avg time|GEMM-only avg time):\s*([0-9.]+)\s*ms\s*\|\s*Throughput:\s*([0-9.]+)\s*GFLOP/s")
PAT_OK    = re.compile(r"Correctness:\s*max_abs_err=([0-9.e+-]+)\s*max_rel_err=([0-9.e+-]+)")

def main():
    if len(sys.argv) != 3:
        print("Usage: parse_to_csv.py <run_txt> <out_csv>")
        sys.exit(1)

    run_txt = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    cur = None
    rows = []
    buf = run_txt.read_text().splitlines()

    for line in buf:
        m = PAT_BENCH.match(line)
        if m:
            cur = {"bench": m.group(1), "avg_ms": "", "gflops": "", "max_abs_err": "", "max_rel_err": ""}
            rows.append(cur)
            continue

        if cur is None:
            continue

        m = PAT_TIME.search(line)
        if m:
            cur["avg_ms"] = m.group(2)
            cur["gflops"] = m.group(3)
            continue

        m = PAT_OK.search(line)
        if m:
            cur["max_abs_err"] = m.group(1)
            cur["max_rel_err"] = m.group(2)
            continue

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("bench,avg_ms,gflops,max_abs_err,max_rel_err\n")
        for r in rows:
            f.write(f'{r["bench"]},{r["avg_ms"]},{r["gflops"]},{r["max_abs_err"]},{r["max_rel_err"]}\n')

    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    main()
