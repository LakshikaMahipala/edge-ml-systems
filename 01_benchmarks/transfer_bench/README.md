Transfer Bench 

Goal
Measure host<->device transfer overhead and jitter.

Why
Transfer time often dominates end-to-end latency at batch=1.

Scripts
- scripts/cuda_memcpy_bench.py
  Benchmarks H2D/D2H for pageable vs pinned memory (p50/p99 + bandwidth)
- scripts/size_to_copy_time.py
  Quick estimate: time ≈ bytes / bandwidth

Run (later)
python scripts/cuda_memcpy_bench.py --device cuda:0 --sizes_mb 1,4,16,64,256 --warmup 50 --iters 200 --out results/memcpy_bench.json
