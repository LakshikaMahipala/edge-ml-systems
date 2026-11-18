# Inference Benchmark Harness

> Reusable tools for measuring ML inference correctly: warm-up, p50/p90/p99 latency, throughput, pre-/post-processing breakdown, host-device transfer cost and queueing effects. Supports PyTorch and ONNX Runtime.

**Status:** 🟡 Code complete · CPU-only tools verified (timer self-test, M/M/1 queue simulation, transfer-budget model) · GPU and model benchmark results pending

## Why
A single "average latency" number hides most of what matters on real systems. This harness separates the stages of an inference request and reports tail latency, which is what users and real-time systems feel.

| Tool | Answers |
|---|---|
| `run_timer_selftest.py` | Is the timer trustworthy (warm-up, percentiles)? |
| `run_pytorch_benchmark.py`, `run_onnxruntime_benchmark.py` | Latency and throughput of a model on CPU or GPU |
| `run_latency_sweep.py` | How latency scales with input size and batch |
| `run_profile_pytorch.py` | Which operators dominate the time? |
| `run_preproc_speedup.py`, `cpp_preproc/` | Python vs C++/SIMD image preprocessing (OpenCV) |
| `run_transfer_budget_demo.py` | When does host↔device copying dominate? |
| `run_queue_sim.py` | Why p99 latency explodes as utilisation approaches 100% (M/M/1 queue) |
| `run_streaming_benchmark.py`, `run_video_decode_benchmark.py` | End-to-end video pipelines |

Example (`run_queue_sim.py`, 20 ms service time, 30 requests/s → 60% utilisation): p50 = 34 ms, **p99 = 250 ms**. Even at 60% load, the slowest 1% of requests wait about 7× longer than the median.

## Quick start

Outputs are written to: `projects/03-inference-benchmark-harness/results/`

```bash
# =========================
# 1) INSTALL
# =========================
pip install -r projects/03-inference-benchmark-harness/requirements.txt

# =========================
# 2) TIMER SELF-TEST
# =========================
python projects/03-inference-benchmark-harness/run_timer_selftest.py

# =========================
# 3) PYTORCH BASELINE BENCHMARK
# =========================
python projects/03-inference-benchmark-harness/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --warmup 20 --iters 100

# =========================
# 4) PYTORCH BENCHMARK + JSON EXPORT
# =========================
python projects/03-inference-benchmark-harness/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --batch 1 --warmup 20 --iters 100 --save_json

# =========================
# 5) PROFILING (OPERATOR-LEVEL)
# =========================
python projects/03-inference-benchmark-harness/run_profile_pytorch.py --model resnet18 --device cpu

# =========================
# 6) MINI-PROJECT A: LATENCY SWEEP
# =========================
python projects/03-inference-benchmark-harness/run_latency_sweep.py --model resnet18 --device cpu --input_sizes 160,224,320 --batches 1,2,4 --save_json

# =========================
# 7) MINI-PROJECT B: QUEUEING / P99 INTUITION
# =========================
python projects/03-inference-benchmark-harness/run_queue_sim.py --service_ms 20 --arrival_rps 30

# =========================
# 8) ACCURACY EVAL (OPTIONAL, NEEDS DATASET ACCESS)
# =========================
python projects/03-inference-benchmark-harness/run_accuracy_eval.py --model resnet18 --device cpu --batch 64 --max_batches 50

# =========================
# 9) PIPELINE DEMO (PRODUCER/CONSUMER)
# =========================
python projects/03-inference-benchmark-harness/run_pipeline_demo.py --model resnet18 --device cpu --queue_size 8 --num_items 200

# =========================
# 10) C++ PREPROCESSING (OPTIONAL, NEEDS EXTENSION BUILD)
# Build instructions:
#   projects/03-inference-benchmark-harness/cpp_preproc/python/README.md
# Demo:
# =========================
python projects/03-inference-benchmark-harness/run_cpp_preproc_demo.py --image path/to/image.jpg --model resnet18 --device cpu

# =========================
# 11) MINI-PROJECT 1A: PYTHON vs C++ PREPROCESS SPEEDUP (OPTIONAL)
# =========================
python projects/03-inference-benchmark-harness/run_preproc_speedup.py --model resnet18 --device cpu --input_size 224 --batch 1 --warmup 20 --iters 100 --image path/to/image.jpg --save_json

# =========================
# 12) FILE I/O BENCHMARK (EDGE REALISM)
# =========================
python projects/03-inference-benchmark-harness/run_file_io_benchmark.py --path path/to/large_file.bin --iters 30 --chunk_kb 256
# Linux optional (may require permissions):
python projects/03-inference-benchmark-harness/run_file_io_benchmark.py --path path/to/large_file.bin --iters 10 --chunk_kb 256 --drop_caches

# =========================
# 13) TRANSFER BUDGET + SOURCE SMOKETESTS
# =========================
python projects/03-inference-benchmark-harness/run_transfer_budget_demo.py --n 1 --c 3 --h 224 --w 224 --dtype_bytes 4
python projects/03-inference-benchmark-harness/run_source_smoketest.py --image_folder path/to/images --max_frames 10
python projects/03-inference-benchmark-harness/run_source_smoketest.py --video path/to/video.mp4 --max_frames 10

# =========================
# 14) STREAMING BENCHMARK (SOURCE -> QUEUE -> CONSUMER)
# =========================
# Closed-loop (backpressure protects p99)
python projects/03-inference-benchmark-harness/run_streaming_benchmark.py --image_folder path/to/images --max_items 200 --queue_size 8 --open_loop_fps 0
# Open-loop (simulate camera FPS)
python projects/03-inference-benchmark-harness/run_streaming_benchmark.py --image_folder path/to/images --max_items 500 --queue_size 8 --open_loop_fps 30
python projects/03-inference-benchmark-harness/run_streaming_benchmark.py --image_folder path/to/images --max_items 500 --queue_size 8 --open_loop_fps 30 --drop_when_full

# =========================
# 15) VIDEO DECODE BENCHMARK
# =========================
python projects/03-inference-benchmark-harness/run_video_decode_benchmark.py --video path/to/video.mp4 --warmup 20 --frames 200

# =========================
# 16) ONNX RUNTIME BENCHMARK (MINI-PROJECT 1B)
# =========================
python projects/03-inference-benchmark-harness/run_onnxruntime_benchmark.py --model resnet18 --input_size 224 --batch 1 --warmup 20 --iters 100 --save_json
# Compare against PyTorch
python projects/03-inference-benchmark-harness/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --warmup 20 --iters 100 --topk 5 --save_json
