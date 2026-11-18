Metrics & Measurement 

Purpose
This document is the single “how we measure” + “how we report” reference for the repository.
It connects:
- measurement methodology
- scripts to run
- what to record
- where artifacts live (JSON, metrics table, plots)

1) Golden rules
- Always warm up before measuring.
- Always report p50 and p99 (not only average).
- Always disclose configuration (device, batch, input size, torch version).
- End-to-end must be explicitly defined (pre + infer + post in this repo).
- Keep representative JSON outputs to support claims.

2) What metrics mean (interpretation)
Latency 
- p50: typical runtime 
- p99: tail runtime (stability / worst-case behavior) 
If p99 >> p50, investigate: 
- OS jitter, thread scheduling 
- memory allocation/fragmentation 
- background processes 
- queueing effects under load 

Accuracy
- Top-1: correct if best prediction matches label 
- Top-5: correct if label appears in top 5 predictions    
Accuracy is meaningless without stating:                  
- dataset and preprocessing                  
- evaluation subset size (n)

Speedup Study (Mini-Project 1A)

- python inference_bench/run_preproc_speedup.py --model resnet18 --device cpu --input_size 224 --batch 1 --warmup 20 --iters 100 --image path/to/image.jpg --save_json
  
Produces:
- Path A (Python preprocess) PerfSummary
- Path B (C++ preprocess) PerfSummary
- Optional JSON outputs for both paths
  
Record:
- preprocess p50/p99 and end-to-end p50/p99 for both paths
- compute speedup ratios

                              

3) Where truth lives (repo policy)
- docs/metrics.md: human-readable summary table (single source of truth)
- inference_bench/results/: machine-readable JSON outputs from --save_json
- docs/plots/: generated figures (committed when they support a milestone)

4) Scripts and what they produce

Timer self-test (sanity check)
- python inference_bench/run_timer_selftest.py
Produces:
- p50/p99 on two CPU workloads

PyTorch benchmark baseline
- python inference_bench/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --batch 1 --warmup 20 --iters 100 --save_json
Produces:
- preprocess/inference/postprocess/end-to-end p50/p99
- system info and peak RSS
- JSON in inference_bench/results/

Profiling (operator hotspots)
- python inference_bench/run_profile_pytorch.py --model resnet18 --device cpu --input_size 224
Produces:
- torch.profiler table (top ops by self time)

Accuracy evaluation (top-1/top-5)
- python inference_bench/run_accuracy_eval.py --model resnet18 --device cpu --batch 64 --max_batches 50
Produces:
- top1/top5 and n

Mini-tools: 
- python inference_bench/run_latency_sweep.py --model resnet18 --device cpu --input_sizes 160,224,320 --batches 1,2,4 --save_json
- python inference_bench/run_queue_sim.py --service_ms 20 --arrival_rps 30

How to update docs/metrics.md 
Option A: manually copy from console outputs
Option B: use helper script for JSON:
- python scripts/update_metrics_from_json.py path/to/result.json
Then paste the markdown row into docs/metrics.md.

Plots (once JSON exists)
- python scripts/plot_results.py --results_dir inference_bench/results --out_dir docs/plots
This will generate:
- p50 vs p99 plot
- latency vs batch/input (if sweep results exist)

Definition of done
Complete when docs/metrics.md has at least:
- one row with real p50/p99 end-to-end numbers
- one row with real top1/top5 from evaluation
and docs/plots contains at least one plot generated from JSON.
