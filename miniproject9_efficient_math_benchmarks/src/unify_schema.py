from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

def add_rows(out, technique, df, make_exp_id, method_col, p50, p99, mean, err_col):
    for _, r in df.iterrows():
        out.append({
            "technique": technique,
            "experiment_id": make_exp_id(r),
            "method": str(r.get(method_col, "")),
            "p50_ms": float(r.get(p50, 0.0)),
            "p99_ms": float(r.get(p99, 0.0)),
            "mean_ms": float(r.get(mean, 0.0)),
            "error": float(r.get(err_col, 0.0)),
            "notes": "",
        })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strassen", type=str, required=True)
    ap.add_argument("--winograd", type=str, required=True)
    ap.add_argument("--fft", type=str, required=True)
    ap.add_argument("--svd", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/unified_results.csv")
    args = ap.parse_args()

    out = []

    # Strassen schema: N, method, leaf_size, p50_ms, p99_ms, mean_ms, max_abs_error
    s = pd.read_csv(args.strassen)
    add_rows(
        out, "strassen", s,
        make_exp_id=lambda r: f"N={int(r['N'])}_leaf={r['leaf_size']}",
        method_col="method", p50="p50_ms", p99="p99_ms", mean="mean_ms", err_col="max_abs_error"
    )

    # Winograd schema: H,W,method,p50_ms,p99_ms,mean_ms,max_abs_error
    w = pd.read_csv(args.winograd)
    add_rows(
        out, "winograd", w,
        make_exp_id=lambda r: f"H={int(r['H'])}_W={int(r['W'])}",
        method_col="method", p50="p50_ms", p99="p99_ms", mean="mean_ms", err_col="max_abs_error"
    )

    # FFT schema: N,K,method,p50_ms,p99_ms,mean_ms,max_abs_error
    f = pd.read_csv(args.fft)
    add_rows(
        out, "fft_conv", f,
        make_exp_id=lambda r: f"N={int(r['N'])}_K={int(r['K'])}",
        method_col="method", p50="p50_ms", p99="p99_ms", mean="mean_ms", err_col="max_abs_error"
    )

    # SVD schema: in_dim,out_dim,batch,rank,method,p50_ms,p99_ms,mean_ms,max_abs_err,rel_err
    sv = pd.read_csv(args.svd)
    # use rel_err as error if present
    for _, r in sv.iterrows():
        out.append({
            "technique": "low_rank_svd",
            "experiment_id": f"in={int(r['in_dim'])}_out={int(r['out_dim'])}_b={int(r['batch'])}_r={r['rank']}",
            "method": str(r.get("method", "")),
            "p50_ms": float(r.get("p50_ms", 0.0)),
            "p99_ms": float(r.get("p99_ms", 0.0)),
            "mean_ms": float(r.get("mean_ms", 0.0)),
            "error": float(r.get("rel_err", r.get("max_abs_err", 0.0))),
            "notes": "",
        })

    out_df = pd.DataFrame(out)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p, index=False)
    print("Wrote:", out_p)

if __name__ == "__main__":
    main()
