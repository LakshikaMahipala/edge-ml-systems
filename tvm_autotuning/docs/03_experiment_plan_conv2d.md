Conv2D tuning experiment plan 

Goal
Tune a single conv2d workload and show measurable speedup.

Target shapes (example)
- N=1, H=W=224, Cin=3, Cout=64, K=7, stride=2 (ResNet stem)
- N=1, H=W=56, Cin=64, Cout=64, K=3, stride=1 (ResNet block)

Procedure
1) Baseline: compile without tuning, measure latency
2) Run tuning for a fixed budget (e.g., 200 trials)
3) Compile with best schedule and re-measure
4) Report speedup and stability (p50/p99)

What to report
- shape parameters
- trials
- tuning time
- baseline latency (p50/p99)
- tuned latency (p50/p99)
- speedup
- logs paths
