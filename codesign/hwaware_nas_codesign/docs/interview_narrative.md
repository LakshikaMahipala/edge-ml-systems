# HW-aware NAS + Co-design

## Problem
Neural Architecture Search (NAS) can generate many model candidates, but in real deployment
we cannot choose models based on accuracy alone. On edge/FPGA targets, latency, energy,
and memory constraints often dominate.

So the problem is:
**How do we select architectures that are both accurate and hardware-feasible without measuring every candidate on hardware?**

## Method
I built a hardware-aware NAS and co-design workflow with an estimator ladder:

1) Multi-objective formulation
- Treat model selection as a Pareto problem over (accuracy, latency, energy).

2) Latency estimation pipeline
- Define a graph representation and feature spec for a GNN latency predictor.
- Implement baseline predictors first:
  - regression MLP on aggregated graph features
  - pairwise ranking model (few-shot friendly) to learn ordering of latency.

3) HW/SW co-design loop
- Define a joint search space (architecture × hardware knobs):
  - Architecture knobs: block type, width, depth
  - Hardware knobs: FPGA parallelism (unroll factor P), tiling, II_factor proxy
- Enumerate candidates and compute proxy objective values.
- Pareto-filter the candidates and select 3 representative designs:
  - fast, balanced, accurate

4) Evidence policy
- All numbers are explicitly labeled as proxies until calibrated with real FPGA measurements.

## Result (current state)
- Delivered a complete, reproducible codebase implementing:
  - Pareto frontier extraction
  - latency predictors (regression + ranking)
  - co-design enumeration and Pareto point selection
- Produced a Mini-project report (Mini-project 13) that ties the entire pipeline together.

## Next measurable step
When FPGA access is available:
- Collect 5–10 real kernel latency measurements.
- Calibrate the proxy latency estimator.
- Replace acc_proxy with reduced-training or real accuracy.
- Re-run the co-design loop and update Pareto points with measured evidence.
