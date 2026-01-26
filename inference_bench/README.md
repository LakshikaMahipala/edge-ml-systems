**** How to Run 
1) Install dependencies
   - See requirements.txt
2) Run timer self-test
   - run_timer_selftest.py
3) Run benchmark
   - run_benchmark.py

Outputs
- results/ will contain benchmark outputs (latency percentiles, etc.)


**** PyTorch Benchmark (real model)

Install (example)
- pip install -r inference_bench/requirements.txt

Run
- python inference_bench/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --warmup 20 --iters 100

What to record into docs/metrics.md
- End-to-end p50 (ms)
- End-to-end p99 (ms)
- Device (CPU/GPU name)
- Torch version + platform string printed by the script

**** Profiling
- Breakdown timing + JSON export:
  python inference_bench/run_pytorch_benchmark.py --model resnet18 --device cpu --save_json
- Operator-level profiler:
  python inference_bench/run_profile_pytorch.py --model resnet18 --device cpu

**** PyTorch benchmark supports:
- --batch (batch size)
- prints system info
- prints peak RSS memory (CPU)

Example (run):
python inference_bench/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --batch 1 --iters 100 --save_json

Mini-project A (latency sweep):
python inference_bench/run_latency_sweep.py --model resnet18 --device cpu --input_sizes 160,224,320 --batches 1,2,4 --save_json

Mini-project B (queueing / p99 intuition):
python inference_bench/run_queue_sim.py --service_ms 20 --arrival_rps 30

**** Documentation
- Measurement standard: docs/methodology_measurement.md
- Mini-Project 0 report: docs/mini_project_0_report.md
- Embedded constraints checklist: docs/embedded_constraints_checklist.md
- Metrics table: docs/metrics.md
- Daily log: docs/daily_log.md

**** Accuracy Evaluation (Top-1 / Top-5)

Goal
Compute top-1 and top-5 accuracy with a reproducible evaluation loop.

Run (later, locally)
python inference_bench/run_accuracy_eval.py --model resnet18 --device cpu --batch 64 --max_batches 50

What to record
- top1, top5, n
- system info lines printed by the script

**** Pipeline demo (Producer/Consumer)

Purpose
Introduce backpressure and concurrency structure used in real-time inference systems.

Run 
python inference_bench/run_pipeline_demo.py --model resnet18 --device cpu --queue_size 8 --num_items 200


**** C++ preprocessing via pybind11

Goal
Call the C++ preprocessing pipeline from Python to prepare for benchmarking speedups.

Build extension
See: cpp_preproc/python/README.md

Demo 
python inference_bench/run_cpp_preproc_demo.py --image path/to/image.jpg --model resnet18 --device cpu

**** Mini-Project 1A — Python vs C++ preprocessing speedup

Benchmark 
python inference_bench/run_preproc_speedup.py --model resnet18 --device cpu --input_size 224 --batch 1 --warmup 20 --iters 100 --image path/to/image.jpg --save_json

What to record
- preprocess p50/p99 for both paths
- end-to-end p50/p99 for both paths
- speedup numbers (Python/C++)
- save JSON and update docs/metrics.md


**** File I/O benchmark (edge realism)

Run
python inference_bench/run_file_io_benchmark.py --path path/to/large_file.bin --iters 30 --chunk_kb 256

Optional cold-cache attempt (Linux, may require permissions)
python inference_bench/run_file_io_benchmark.py --path path/to/large_file.bin --iters 10 --chunk_kb 256 --drop_caches

Record
- p50, p99, mean read latency
- throughput (MB/s)
- whether cache dropping was possible

**** Transfer budget model + input sources

Transfer budget demo (later)
python inference_bench/run_transfer_budget_demo.py --n 1 --c 3 --h 224 --w 224 --dtype_bytes 4

Source smoketest (later)
python inference_bench/run_source_smoketest.py --image_folder path/to/images --max_frames 10
python inference_bench/run_source_smoketest.py --video path/to/video.mp4 --max_frames 10

**** Streaming benchmark (source -> queue -> consumer)

Closed-loop (backpressure, protects p99)
python inference_bench/run_streaming_benchmark.py --image_folder path/to/images --max_items 200 --queue_size 8 --open_loop_fps 0

Open-loop (simulate camera FPS, may queue or drop)
python inference_bench/run_streaming_benchmark.py --image_folder path/to/images --max_items 500 --queue_size 8 --open_loop_fps 30
python inference_bench/run_streaming_benchmark.py --image_folder path/to/images --max_items 500 --queue_size 8 --open_loop_fps 30 --drop_when_full

Record
- throughput
- e2e p50/p99
- queue wait p50/p99
- service p50/p99

**** Camera/video stack — decode benchmark

Run (later)
python inference_bench/run_video_decode_benchmark.py --video path/to/video.mp4 --warmup 20 --frames 200

Record
- decode p50/p99 (ms)
- approx decode FPS
