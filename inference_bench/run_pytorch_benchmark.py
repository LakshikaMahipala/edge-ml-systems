# inference_bench/run_pytorch_benchmark.py
from __future__ import annotations

import argparse
import platform
import sys

import torch
from inference_bench.src.system_info import get_system_info
from inference_bench.src.memory import get_peak_rss_mb
from inference_bench.src.timer import Timer
from inference_bench.src.pytorch_infer import PyTorchConfig, PyTorchRunner
from inference_bench.src.reporting import RunMeta, PerfSummary, print_budget, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"],
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument(
        "--save_json",
        action="store_true",
        help="Save results JSON into inference_bench/results/",
    )
    args = ap.parse_args()

    cfg = PyTorchConfig(
        model_name=args.model,
        device=args.device,
        input_size=args.input_size,
        topk=args.topk,
        batch=args.batch,
    )
    runner = PyTorchRunner(cfg)

    # For GPU later: ensure accurate timing by syncing before/after
    sync = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        sync = torch.cuda.synchronize

    timer = Timer(warmup_iters=args.warmup, measure_iters=args.iters, sync=sync)

    # Measure components
    r_pre = timer.run(lambda: runner.preprocess())

    x = runner.preprocess()
    r_inf = timer.run(lambda: runner.forward(x))

    logits = runner.forward(x)
    r_post = timer.run(lambda: runner.postprocess(logits))

    # Measure end-to-end
    r_e2e = timer.run(lambda: runner.end_to_end())

    # Show one sample output (not inside timed loops)
    pred_idx, pred_prob = runner.end_to_end()

    print("PyTorch Benchmark (Day 3/Day 4)")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__}")
    print(
        f"Args: model={args.model}, device={args.device}, input_size={args.input_size}, "
        f"warmup={args.warmup}, iters={args.iters}, topk={args.topk}, save_json={args.save_json}"
    )
    print("")

    print("Preprocess:", r_pre.summary())
    print("Inference :", r_inf.summary())
    print("Postproc  :", r_post.summary())
    print("")
    print("End-to-end:", r_e2e.summary())
    print("")
    print(f"Sample top-{args.topk}: {list(zip(pred_idx, [round(p, 6) for p in pred_prob]))}")
    print("")

    sysinfo = get_system_info(args.device)
    print("System Info:", sysinfo.to_dict())
    print("")
    # ---- additions: PerfSummary + budget printing + optional JSON save ----
    perf = PerfSummary(
        preprocess=r_pre.summary(),
        inference=r_inf.summary(),
        postprocess=r_post.summary(),
        end_to_end=r_e2e.summary(),
    )

    print_budget(perf)

    meta = RunMeta(
        project="inference_bench",
        script="run_pytorch_benchmark.py",
        model=args.model,
        device=args.device,
        input_desc=f"1x3x{args.input_size}x{args.input_size}",
        warmup=args.warmup,
        iters=args.iters,
        platform=platform.platform(),
        python=sys.version.split()[0],
        torch=torch.__version__,
    )

    if args.save_json:
        out_path = save_json(meta, perf)
        print(f"Saved JSON: {out_path}")
    mem = get_peak_rss_mb()
    print(f"Peak RSS: {mem.peak_rss_mb:.2f} MB ({mem.note})")

if __name__ == "__main__":
    main()
