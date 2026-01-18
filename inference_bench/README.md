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
