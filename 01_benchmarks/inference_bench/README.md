# Inference Bench — Benchmarking Foundations (Project 0)

Outputs are written to: `01_benchmarks/inference_bench/results/`

```bash
# =========================
# 1) INSTALL
# =========================
pip install -r 01_benchmarks/inference_bench/requirements.txt

# =========================
# 2) TIMER SELF-TEST
# =========================
python 01_benchmarks/inference_bench/run_timer_selftest.py

# =========================
# 3) PYTORCH BASELINE BENCHMARK
# =========================
python 01_benchmarks/inference_bench/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --warmup 20 --iters 100

# =========================
# 4) PYTORCH BENCHMARK + JSON EXPORT
# =========================
python 01_benchmarks/inference_bench/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --batch 1 --warmup 20 --iters 100 --save_json

# =========================
# 5) PROFILING (OPERATOR-LEVEL)
# =========================
python 01_benchmarks/inference_bench/run_profile_pytorch.py --model resnet18 --device cpu

# =========================
# 6) MINI-PROJECT A: LATENCY SWEEP
# =========================
python 01_benchmarks/inference_bench/run_latency_sweep.py --model resnet18 --device cpu --input_sizes 160,224,320 --batches 1,2,4 --save_json

# =========================
# 7) MINI-PROJECT B: QUEUEING / P99 INTUITION
# =========================
python 01_benchmarks/inference_bench/run_queue_sim.py --service_ms 20 --arrival_rps 30

# =========================
# 8) ACCURACY EVAL (OPTIONAL, NEEDS DATASET ACCESS)
# =========================
python 01_benchmarks/inference_bench/run_accuracy_eval.py --model resnet18 --device cpu --batch 64 --max_batches 50

# =========================
# 9) PIPELINE DEMO (PRODUCER/CONSUMER)
# =========================
python 01_benchmarks/inference_bench/run_pipeline_demo.py --model resnet18 --device cpu --queue_size 8 --num_items 200

# =========================
# 10) C++ PREPROCESSING (OPTIONAL, NEEDS EXTENSION BUILD)
# Build instructions:
#   01_benchmarks/inference_bench/cpp_preproc/python/README.md
# Demo:
# =========================
python 01_benchmarks/inference_bench/run_cpp_preproc_demo.py --image path/to/image.jpg --model resnet18 --device cpu

# =========================
# 11) MINI-PROJECT 1A: PYTHON vs C++ PREPROCESS SPEEDUP (OPTIONAL)
# =========================
python 01_benchmarks/inference_bench/run_preproc_speedup.py --model resnet18 --device cpu --input_size 224 --batch 1 --warmup 20 --iters 100 --image path/to/image.jpg --save_json

# =========================
# 12) FILE I/O BENCHMARK (EDGE REALISM)
# =========================
python 01_benchmarks/inference_bench/run_file_io_benchmark.py --path path/to/large_file.bin --iters 30 --chunk_kb 256
# Linux optional (may require permissions):
python 01_benchmarks/inference_bench/run_file_io_benchmark.py --path path/to/large_file.bin --iters 10 --chunk_kb 256 --drop_caches

# =========================
# 13) TRANSFER BUDGET + SOURCE SMOKETESTS
# =========================
python 01_benchmarks/inference_bench/run_transfer_budget_demo.py --n 1 --c 3 --h 224 --w 224 --dtype_bytes 4
python 01_benchmarks/inference_bench/run_source_smoketest.py --image_folder path/to/images --max_frames 10
python 01_benchmarks/inference_bench/run_source_smoketest.py --video path/to/video.mp4 --max_frames 10

# =========================
# 14) STREAMING BENCHMARK (SOURCE -> QUEUE -> CONSUMER)
# =========================
# Closed-loop (backpressure protects p99)
python 01_benchmarks/inference_bench/run_streaming_benchmark.py --image_folder path/to/images --max_items 200 --queue_size 8 --open_loop_fps 0
# Open-loop (simulate camera FPS)
python 01_benchmarks/inference_bench/run_streaming_benchmark.py --image_folder path/to/images --max_items 500 --queue_size 8 --open_loop_fps 30
python 01_benchmarks/inference_bench/run_streaming_benchmark.py --image_folder path/to/images --max_items 500 --queue_size 8 --open_loop_fps 30 --drop_when_full

# =========================
# 15) VIDEO DECODE BENCHMARK
# =========================
python 01_benchmarks/inference_bench/run_video_decode_benchmark.py --video path/to/video.mp4 --warmup 20 --frames 200

# =========================
# 16) ONNX RUNTIME BENCHMARK (MINI-PROJECT 1B)
# =========================
python 01_benchmarks/inference_bench/run_onnxruntime_benchmark.py --model resnet18 --input_size 224 --batch 1 --warmup 20 --iters 100 --save_json
# Compare against PyTorch
python 01_benchmarks/inference_bench/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --warmup 20 --iters 100 --topk 5 --save_json
