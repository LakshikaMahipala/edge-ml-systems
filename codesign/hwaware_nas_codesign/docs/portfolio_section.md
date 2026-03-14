# Portfolio section: HW-aware NAS + Co-design 

## What it is
A complete workflow for selecting neural architectures under deployment constraints,
with an estimator ladder that enables scaling NAS decisions before full hardware measurement.

## Modules
- pareto_week15/: Pareto dominance + frontier extraction
- gnn_latency_week15/: GNN latency predictor design spec (features/labels/protocol)
- latency_predictor_week15/: baseline latency predictors (regression + pairwise ranking)
- codesign_week15/: joint architecture × hardware co-design loop + Pareto point selection
- miniproject13_hwaware_nas_codesign/: consolidated report tying the workflow together

## Evidence policy
All reported latencies/energies are proxies until calibrated with real FPGA measurements.
The repo includes a clear run-later plan to replace proxies with measured numbers.

## Why it matters
This is the core skill for ML hardware roles:
choosing models that meet real-time constraints and proving decisions with measurable evidence.
