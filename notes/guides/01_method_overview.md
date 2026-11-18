# Method overview

We implement the standard loop:

1) Candidate generation
- CPU/GPU: tiling candidates from tvm_autotuning Day 2
- FPGA: sweep points from fpga_tiling_sweep Day 5

2) Feature extraction
- shape + tiling params + derived features (bytes, macs, intensity)
- categorical: op_type / backend (cpu/fpga)

3) Train predictor
- baseline: tree model (sklearn)
- optional: MLP

4) Rank candidates
- predict latency for each candidate
- rank ascending (lowest predicted latency best)

5) Select
- choose top-k (k=5,10,20)
- (later) actually measure those k candidates
- pick the best measured

Key evaluation metrics
- Spearman rank correlation between predicted and true latency
- regret@k = (best_true - best_true_in_topk)
