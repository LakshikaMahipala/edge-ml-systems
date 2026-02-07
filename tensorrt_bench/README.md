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

FP16 build + benchmark 

1) Build FP16 engine
python scripts/build_engine.py --onnx results/model.onnx --engine results/resnet18_fp16.plan --workspace_mb 2048 --fp16

2) Run benchmark and save log
python scripts/run_trt_benchmark.py --engine results/resnet18_fp16.plan --warmup 200 --iters 1000 --batch 1 --log results/resnet18_fp16_trtexec.txt --run

3) Parse to JSON (p50/p99)
python scripts/parse_trtexec_log.py --log results/resnet18_fp16_trtexec.txt --out results/resnet18_fp16_summary.json --model resnet18 --precision fp16 --batch 1


INT8 PTQ (Week 5 Day 5)

1) Create calibration tensor (placeholder data)
python scripts/make_calib_data.py --out results/calib/calib_fp32.npy --n 256 --batch 16 --input_size 224

2) Build FP32 engine (existing workflow)
python scripts/build_engine.py --onnx results/model.onnx --engine results/resnet18_fp32.plan --workspace_mb 2048 --run

3) Build INT8 engine + calibration cache
python scripts/build_int8_engine.py --onnx results/model.onnx --engine results/resnet18_int8.plan \
  --calib_npy results/calib/calib_fp32.npy --calib_cache results/calib/resnet18.cache --batch 16 --workspace_mb 2048

4) Compare FP32 vs INT8 (accuracy proxy)
python scripts/compare_fp32_int8.py --fp32_engine results/resnet18_fp32.plan --int8_engine results/resnet18_int8.plan \
  --calib_npy results/calib/calib_fp32.npy --n 128 --batch 16

Important
- For real accuracy delta, replace FakeData with a real dataset matching the model’s label space (ImageNet val for torchvision ImageNet models).
