Measurement protocol (so your tuning results are credible)

Warmup
- run several warmup iterations before timing

Statistics
- report p50 and p99 latency (not just mean)
- record number of runs (iters)

Environment logging
- CPU model / GPU model
- OS
- Python + TVM versions
- thread settings (OMP_NUM_THREADS)
- clock settings if possible

Stability
- avoid running other heavy apps
- run multiple times to confirm the best schedule is reproducible

Artifacts to save
- tuning log file (TVM trace)
- best schedule config JSON
- summary table row (CSV)
