TensorRT Bench 

Goal
- Build a TensorRT FP32 baseline engine from ONNX.
- Benchmark latency/throughput in a repeatable way.

Why we use trtexec first
- Standard NVIDIA tool
- Minimizes Python API version issues
- Produces comparable performance numbers

Workflow (run later on an NVIDIA machine)

1) Export ONNX
python scripts/export_onnx.py --model resnet18 --out results/model.onnx --input_size 224

2) Build FP32 engine
python scripts/build_engine.py --onnx results/model.onnx --engine results/resnet18_fp32.plan --workspace_mb 1024

Then run the printed trtexec command.

3) Benchmark FP32 engine
python scripts/run_trt_benchmark.py --engine results/resnet18_fp32.plan --warmup 200 --iters 1000 --batch 1

Outputs to record into docs/metrics.md
- model, precision, batch
- latency: mean, p50, p99
- throughput (inf/s)
- build settings: workspace MB, TensorRT version, GPU name
