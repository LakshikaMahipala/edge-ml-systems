# inference_bench/run_pipeline_demo.py
from __future__ import annotations

import argparse
import time

import torch

from inference_bench.src.pipeline import PipelineConfig, ProducerConsumerPipeline
from inference_bench.src.pytorch_infer import PyTorchConfig, PyTorchRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--queue_size", type=int, default=8)
    ap.add_argument("--num_items", type=int, default=200)
    ap.add_argument("--producer_sleep_ms", type=float, default=0.0)
    ap.add_argument("--consumer_sleep_ms", type=float, default=0.0)
    args = ap.parse_args()

    runner = PyTorchRunner(
        PyTorchConfig(
            model_name=args.model,
            device=args.device,
            input_size=args.input_size,
            batch=args.batch,
        )
    )

    # Producer: generate a dummy "frame id"
    def producer_fn(i: int):
        # In real systems, this would be: read frame -> preprocess.
        # For now, we just pass an integer token to show the pipeline structure.
        return i

    # Consumer: run end-to-end inference using preallocated tensor inside runner
    def consumer_fn(_item):
        # In real systems, consumer would accept a preprocessed tensor from queue.
        # We keep it simple today: run runner.end_to_end() which includes pre/infer/post.
        return runner.end_to_end()

    cfg = PipelineConfig(
        queue_size=args.queue_size,
        num_items=args.num_items,
        producer_sleep_ms=args.producer_sleep_ms,
        consumer_sleep_ms=args.consumer_sleep_ms,
    )

    pipe = ProducerConsumerPipeline(cfg=cfg, producer_fn=producer_fn, consumer_fn=consumer_fn)

    t0 = time.time()
    pipe.start()
    pipe.join()
    t1 = time.time()

    dt = t1 - t0
    produced = pipe.stats.produced
    consumed = pipe.stats.consumed
    throughput = consumed / dt if dt > 0 else 0.0

    print("Pipeline Demo (Week 2 Day 3)")
    print(f"Produced: {produced}, Consumed: {consumed}, Wall time: {dt:.3f}s, Throughput: {throughput:.2f} items/s")
    print("Note: This demo uses runner.end_to_end() in the consumer. Next steps will pass preprocessed tensors through the queue.")


if __name__ == "__main__":
    main()
