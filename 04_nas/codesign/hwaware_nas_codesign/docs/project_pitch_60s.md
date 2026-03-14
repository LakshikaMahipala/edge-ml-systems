# 60-second pitch 

I built a hardware-aware NAS and co-design pipeline for selecting models under deployment constraints.
Instead of optimizing accuracy alone, I formulated selection as a multi-objective Pareto problem over
accuracy, latency, and energy.

To scale evaluation without benchmarking every model, I designed a graph-based latency predictor
spec and implemented strong baselines: an MLP regressor and a pairwise ranking model for few-shot
latency ordering.

Then I built a toy co-design loop that jointly searches architecture knobs (block/width/depth) and FPGA
hardware knobs (parallelism, tiling, pipelining proxy), computes proxy costs, and extracts Pareto-optimal
design points (fast/balanced/accurate).

The repo is structured with an estimator ladder and strict evidence policy, so it’s ready to be calibrated
with a small set of real FPGA measurements and then used for hardware-aware NAS decisions.
