from __future__ import annotations
import argparse
from pathlib import Path
import json
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results/run1")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--target", type=str, default="llvm")  # cpu
    args = ap.parse_args()

    # NOTE: to run later you need tvm installed
    import tvm
    from tvm import te, auto_scheduler

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Example: conv2d NCHW
    N, Cin, H, W = 1, 64, 56, 56
    Cout, KH, KW = 64, 3, 3
    stride, padding = 1, 1

    data = te.placeholder((N, Cin, H, W), name="data")
    weight = te.placeholder((Cout, Cin, KH, KW), name="weight")
    conv = te.compute(
        (N, Cout, H, W),
        lambda n, co, y, x: te.sum(
            data[n, te.reduce_axis((0, Cin), "ci"), y, x] * weight[co, te.reduce_axis((0, Cin), "ci"), 0, 0],
            axis=[],
        ),
        name="conv",
    )

    # The above is intentionally minimal placeholder.
    # When you run, replace with TOPI conv2d workload (more realistic).

    task = auto_scheduler.SearchTask(func=None, args=(), target=args.target)

    log_file = out_dir / "tuning_log.json"

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=args.trials,
        measure_callbacks=[auto_scheduler.RecordToFile(str(log_file))],
        verbose=2,
    )

    t0 = time.time()
    task.tune(tune_option)
    tune_time = time.time() - t0

    best_sch, best_args = task.apply_best(str(log_file))
    print("Tuning done. Time:", tune_time)
    print("Log file:", log_file)

    meta = {"trials": args.trials, "target": args.target, "tune_time_sec": tune_time, "log_file": str(log_file)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
