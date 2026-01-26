#!/usr/bin/env bash
set -e

echo "Container ready."
echo "PYTHONPATH=$PYTHONPATH"
echo ""
echo "Examples:"
echo "  python inference_bench/run_preproc_speedup.py --model resnet18 --device cpu --iters 50 --warmup 10 --image /workspace/data/sample.jpg"
echo "  python inference_bench/run_file_io_benchmark.py --path /workspace/data/large.bin --iters 20"
echo ""

exec "$@"
