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


