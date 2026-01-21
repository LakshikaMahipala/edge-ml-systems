# inference_bench/run_latency_sweep.py
from __future__ import annotations

import argparse

from inference_bench.src.timer import Timer
from inference_bench.src.pytorch_infer import PyTorchConfig, PyTorchRunner
from inference_bench.src.reporting import RunMeta, PerfSummary, save_json
from inference_bench.src.system_info import get_system_info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--input_sizes", type=str, default="160,224,320")
    ap.add_argument("--batches", type=str, default="1,2,4,8")
    args = ap.parse_args()

    input_sizes = [int(x.strip()) for x in args.input_sizes.split(",") if x.strip()]
    batches = [int(x.strip()) for x in args.batches.split(",") if x.strip()]

    sysinfo = get_system_info(args.device).to_dict()
    print("System Info:", sysinfo)

    timer = Timer(warmup_iters=args.warmup, measure_iters=args.iters)

    for inp in input_sizes:
        for b in batches:
            cfg = PyTorchConfig(model_name=args.model, device=args.device, input_size=inp, batch=b)
            runner = PyTorchRunner(cfg)

            r_e2e = timer.run(lambda: runner.end_to_end())

            perf = PerfSummary(
                preprocess={"p50_ms": float("nan"), "p90_ms": float("nan"), "p99_ms": float("nan"), "min_ms": float("nan"), "max_ms": float("nan")},
                inference={"p50_ms": float("nan"), "p90_ms": float("nan"), "p99_ms": float("nan"), "min_ms": float("nan"), "max_ms": float("nan")},
                postprocess={"p50_ms": float("nan"), "p90_ms": float("nan"), "p99_ms": float("nan"), "min_ms": float("nan"), "max_ms": float("nan")},
                end_to_end=r_e2e.summary(),
            )

            meta = RunMeta(
                project="inference_bench_sweep",
                script="run_latency_sweep.py",
                model=args.model,
                device=args.device,
                input_desc=f"{b}x3x{inp}x{inp}",
                warmup=args.warmup,
                iters=args.iters,
                platform=sysinfo["platform"],
                python=sysinfo["python"],
                torch=sysinfo["torch"],
            )

            print(f"[sweep] model={args.model} device={args.device} input={inp} batch={b} -> {r_e2e.summary()}")

            if args.save_json:
                out = save_json(meta, perf)
                print(f"  saved: {out}")


if __name__ == "__main__":
    main()
