TensorRT Optimization Playbook (Week 5)

Purpose
This playbook is the practical guide for turning a PyTorch model into a TensorRT engine and measuring real latency (p50/p99).
It also documents FP16 and INT8 (PTQ) workflows and the common pitfalls.

Core pipeline (always)
1) PyTorch -> ONNX export
2) ONNX -> TensorRT engine build
3) Engine -> benchmark (warmup + iterations)
4) Parse logs -> JSON -> report tables

Why warmup matters
- GPU clocks ramp up.
- CUDA kernels allocate/initialize lazily.
- First-iteration timings are not representative.
Rule: always run warmup and discard those iterations.

What metrics we care about
- p50 latency: typical user experience
- p99 latency: tail latency (systems reliability)
- Throughput (qps): useful but can hide tail latency problems

Backend comparison philosophy
- PyTorch eager is the baseline (simple, flexible)
- ONNX Runtime is a strong mid-ground (graph runtime)
- TensorRT is usually the performance ceiling on NVIDIA GPUs (engine + fusions + tactics)

--------------------------------------------
FP32 Baseline
--------------------------------------------
1) Export ONNX
python scripts/export_onnx.py --model resnet18 --out results/model.onnx --input_size 224

2) Build FP32 engine (trtexec)
python scripts/build_engine.py --onnx results/model.onnx --engine results/resnet18_fp32.plan --workspace_mb 2048

3) Benchmark + save log
python scripts/run_trt_benchmark.py --engine results/resnet18_fp32.plan --warmup 200 --iters 1000 --batch 1 \
  --log results/resnet18_fp32_trtexec.txt --run

4) Parse log -> JSON
python scripts/parse_trtexec_log.py --log results/resnet18_fp32_trtexec.txt \
  --out results/resnet18_fp32_summary.json --model resnet18 --precision fp32 --batch 1

Record into docs/mini_project_3_trt_vs_ort_vs_pytorch.md

--------------------------------------------
FP16 (speed via lower precision)
--------------------------------------------
1) Build FP16 engine
python scripts/build_engine.py --onnx results/model.onnx --engine results/resnet18_fp16.plan --workspace_mb 2048 --fp16

2) Benchmark + parse (same as FP32)
python scripts/run_trt_benchmark.py --engine results/resnet18_fp16.plan --warmup 200 --iters 1000 --batch 1 \
  --log results/resnet18_fp16_trtexec.txt --run

python scripts/parse_trtexec_log.py --log results/resnet18_fp16_trtexec.txt \
  --out results/resnet18_fp16_summary.json --model resnet18 --precision fp16 --batch 1

Expectation
- FP16 often improves latency due to Tensor Core usage (GPU-dependent).
- Real gain varies by model (memory-bound vs compute-bound).

--------------------------------------------
INT8 PTQ (calibration + quantization)
--------------------------------------------
INT8 is not “just 8-bit compute”.
You must estimate activation ranges via calibration.

1) Create calibration dataset tensor (placeholder now)
python scripts/make_calib_data.py --out results/calib/calib_fp32.npy --n 256 --batch 16 --input_size 224

2) Build INT8 engine + write calibration cache
python scripts/build_int8_engine.py --onnx results/model.onnx --engine results/resnet18_int8.plan \
  --calib_npy results/calib/calib_fp32.npy --calib_cache results/calib/resnet18.cache --batch 16 --workspace_mb 2048

3) Benchmark INT8 engine (trtexec)
python scripts/run_trt_benchmark.py --engine results/resnet18_int8.plan --warmup 200 --iters 1000 --batch 1 \
  --log results/resnet18_int8_trtexec.txt --run

python scripts/parse_trtexec_log.py --log results/resnet18_int8_trtexec.txt \
  --out results/resnet18_int8_summary.json --model resnet18 --precision int8 --batch 1

4) Accuracy delta proxy (drift vs FP32)
python scripts/compare_fp32_int8.py --fp32_engine results/resnet18_fp32.plan --int8_engine results/resnet18_int8.plan \
  --calib_npy results/calib/calib_fp32.npy --n 128 --batch 16

Interpretation
- True accuracy delta needs real labeled validation data (e.g., ImageNet val).
- Proxy metrics help detect “quantization broke the model”.

--------------------------------------------
Common pitfalls (read this)
--------------------------------------------
1) Wrong preprocessing
- Calibrator must see the same normalization as real inference.
- If preprocessing differs, calibration ranges will be wrong -> bad INT8 accuracy.

2) Comparing apples to oranges
- Same input tensor shapes
- Same batch size
- Same warmup and iteration counts
- Same device power mode (later)

3) Ignoring copies
- trtexec numbers mostly reflect engine time; full app time includes data movement.
- That’s why inference_bench measures preprocess/postprocess separately.

Where FPGA fits (bridge to our FPGA track)
- TRT optimizes GPU compute.
- FPGA work starts by making I/O and fixed-point arithmetic correct.
- Early FPGA performance is bounded by transfer (UART). Use the latency budget tool:
  python tools/latency_budget.py --t_pre_ms ... (see docs/latency_budget_model.md)
