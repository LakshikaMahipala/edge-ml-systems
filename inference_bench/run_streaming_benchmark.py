# inference_bench/run_streaming_benchmark.py
from __future__ import annotations

import argparse
import platform
import sys
from typing import Any, List

import torch

from inference_bench.src.pipeline import PipelineConfig, ProducerConsumerPipeline
from inference_bench.src.streaming_bench import summarize_streaming
from inference_bench.src.sources import ImageFolderSource, VideoFileSource
from inference_bench.src.pytorch_infer import PyTorchConfig, PyTorchRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max_items", type=int, default=200)
    ap.add_argument("--queue_size", type=int, default=8)
    ap.add_argument("--open_loop_fps", type=float, default=0.0, help="0=backpressure (closed-loop). >0=fixed-rate producer")
    ap.add_argument("--drop_when_full", action="store_true")
    ap.add_argument("--image_folder", type=str, default="")
    ap.add_argument("--video", type=str, default="")
    ap.add_argument("--max_frames", type=int, default=200)
    args = ap.parse_args()

    if not args.image_folder and not args.video:
        raise SystemExit("Provide either --image_folder or --video")

    print("Streaming Benchmark (Week 3 Day 5)")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__}")
    print(f"Args: model={args.model}, device={args.device}, input_size={args.input_size}, batch={args.batch}, max_items={args.max_items}, queue_size={args.queue_size}, open_loop_fps={args.open_loop_fps}, drop_when_full={args.drop_when_full}")
    print("")

    # Source: yields raw frames (BGR uint8)
    if args.image_folder:
        src = ImageFolderSource(folder=args.image_folder, max_frames=args.max_frames)
    else:
        src = VideoFileSource(path=args.video, max_frames=args.max_frames)

    frames = list(src)  # store to simplify; later we can stream without materializing
    if not frames:
        raise SystemExit("No frames loaded from source")

    runner = PyTorchRunner(
        PyTorchConfig(
            model_name=args.model,
            device=args.device,
            input_size=args.input_size,
            batch=args.batch,
        )
    )

    # Producer returns a frame (BGR numpy array)
    def producer_fn(i: int) -> Any:
        return frames[i % len(frames)].bgr

    # Consumer does: preprocess -> forward -> postprocess
    # Today we use runner.end_to_end() for simplicity.
    # Next step (Week 3 Day 6) will pass the actual frame through preprocess explicitly.
    def consumer_fn(_payload: Any) -> Any:
        return runner.end_to_end()

    cfg = PipelineConfig(
        queue_size=args.queue_size,
        max_items=args.max_items,
        open_loop_fps=args.open_loop_fps,
        drop_when_full=args.drop_when_full,
    )

    pipe = ProducerConsumerPipeline(cfg=cfg, producer_fn=producer_fn, consumer_fn=consumer_fn)
    pipe.start()
    pipe.join()

    summary = summarize_streaming(pipe.timings, pipe.stats.wall_time_s)

    print("Results")
    print(f"Items processed: {summary.count}")
    print(f"Throughput: {summary.throughput_items_s:.2f} items/s")
    print("")
    print(f"E2E latency p50/p99: {summary.e2e_p50_ms:.3f} ms / {summary.e2e_p99_ms:.3f} ms")
    print(f"Queue wait   p50/p99: {summary.queue_p50_ms:.3f} ms / {summary.queue_p99_ms:.3f} ms")
    print(f"Service time p50/p99: {summary.service_p50_ms:.3f} ms / {summary.service_p99_ms:.3f} ms")
    print("")
    print("Interpretation")
    print("- If queue p99 is large, producer outruns consumer (queueing dominates).")
    print("- Closed-loop (open_loop_fps=0) protects tail latency via backpressure.")
    print("- Open-loop shows what happens at fixed camera FPS; drops may occur if enabled.")


if __name__ == "__main__":
    main()
