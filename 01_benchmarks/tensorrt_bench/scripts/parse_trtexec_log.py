import argparse
import json
import re
from pathlib import Path

# trtexec output formats vary slightly by version.
# We parse:
# - Throughput: "Throughput: XXX qps"
# - Latency: "Latency: min = X ms, max = Y ms, mean = Z ms, median = A ms, percentile(99%) = B ms"
# Some versions print "percentile(99%)" or "99th percentile".
PAT_THROUGHPUT = re.compile(r"Throughput:\s*([0-9.]+)\s*qps", re.IGNORECASE)
PAT_LAT_BLOCK = re.compile(r"Latency:\s*.*", re.IGNORECASE)
PAT_MEAN = re.compile(r"mean\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE)
PAT_MEDIAN = re.compile(r"median\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE)
PAT_P99 = re.compile(r"(percentile\(99%\)|99th percentile)\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/trt_summary.json")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--precision", type=str, default="")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", type=str, default="gpu")
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="ignore").splitlines()

    throughput = None
    mean_ms = None
    p50_ms = None
    p99_ms = None

    for line in text:
        m = PAT_THROUGHPUT.search(line)
        if m:
            throughput = float(m.group(1))

        if "Latency:" in line:
            m2 = PAT_MEAN.search(line)
            if m2:
                mean_ms = float(m2.group(1))
            m3 = PAT_MEDIAN.search(line)
            if m3:
                p50_ms = float(m3.group(1))
            m4 = PAT_P99.search(line)
            if m4:
                p99_ms = float(m4.group(2))

    out = {
        "meta": {
            "model": args.model,
            "precision": args.precision,
            "batch": args.batch,
            "device": args.device,
        },
        "metrics": {
            "throughput_qps": throughput,
            "latency_mean_ms": mean_ms,
            "latency_p50_ms": p50_ms,
            "latency_p99_ms": p99_ms,
        },
        "source_log": str(args.log),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
