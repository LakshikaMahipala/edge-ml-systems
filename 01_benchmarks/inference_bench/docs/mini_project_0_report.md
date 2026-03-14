Mini-Project 0 Report — Baseline Inference Benchmarking (Timer + PyTorch)


1) Goal
Build a correct benchmarking foundation for ML hardware/systems work:
- warmup + repeated measurement
- percentile reporting (p50/p99)
- component breakdown (pre/infer/post)
- optional JSON export for reproducible tracking
- system info and memory peak reporting

2) Why this matters (systems rationale)
   
In real deployments:
- p50 is not sufficient; p99 defines user experience and stability.
- preprocessing often dominates end-to-end latency (Amdahl bottleneck).
- memory constraints frequently dominate feasibility on embedded devices.

3) What was implemented
   
Core components:
- inference_bench/src/timer.py
  - warmup iterations excluded
  - uses perf_counter
  - reports p50/p90/p99/min/max
- inference_bench/src/pytorch_infer.py
  - standard pretrained model runner (resnet18 baseline)
  - preprocess → forward → postprocess pipeline
  - batch support 
- inference_bench/src/reporting.py
  - prints latency budget
  - saves results JSON into inference_bench/results/
- inference_bench/src/system_info.py
  - prints platform + cpu count + torch version
- inference_bench/src/memory.py
  - peak RSS reporting on Unix-like systems
Scripts:
- inference_bench/run_pytorch_benchmark.py
  - measures preprocess, inference, postprocess, end-to-end
  - prints p50/p99 and budget
  - optionally saves JSON (--save_json)
- inference_bench/run_profile_pytorch.py
  - torch.profiler operator-level breakdown (CPU/CUDA)
- inference_bench/run_latency_sweep.py
  - sweeps batch and input sizes; saves JSON (optional)
- inference_bench/run_queue_sim.py
  - M/M/1 queue simulation to explain p99 growth under load

4) How to run 
Install:
- pip install -r inference_bench/requirements.txt

Benchmark (baseline):
- python inference_bench/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --batch 1 --warmup 20 --iters 100 --save_json

Profiler:
- python inference_bench/run_profile_pytorch.py --model resnet18 --device cpu --input_size 224

Sweep:
- python inference_bench/run_latency_sweep.py --model resnet18 --device cpu --input_sizes 160,224,320 --batches 1,2,4 --save_json

Queueing intuition:
- python inference_bench/run_queue_sim.py --service_ms 20 --arrival_rps 30

5) Results (TBD until executed)
System info (copy from script output):
- Platform: TBD
- Python: TBD
- Torch: TBD
- Device name: TBD

Primary metrics (copy from End-to-end summary):
- End-to-end p50: TBD ms
- End-to-end p99: TBD ms
- Peak RSS: TBD MB

Component budget (copy from breakdown):
- Preprocess p50/p99: TBD / TBD
- Inference  p50/p99: TBD / TBD
- Postproc   p50/p99: TBD / TBD

6) Interpretation checklist (what we will analyze once numbers exist)
- Is preprocess dominating end-to-end?
  - If yes: optimize preprocessing first (C++/SIMD/parallel pipeline).
- Is inference dominating?
  - If yes: consider ONNX Runtime, TensorRT, quantization, operator fusion.
- Is p99 much larger than p50?
  - If yes: suspect OS jitter, allocation, thread scheduling, or queueing effects.
- Does batch increase throughput but inflate p99?
  - If yes: balance batch vs tail latency (deployment tradeoff).

7) Next actions 
- Produce first real metrics row in docs/metrics.md (from JSON output)
- Add a small validation set accuracy computation (top-1/top-5)
- Add ONNX Runtime baseline for comparison (future mini-project)
- Start building a repeatable CI-based “smoke benchmark” (prints system info and runs quick timing)
