# Mini-project 8 Report (Predictor-based Schedule Selection)

## Objective
Use a predictor to rank many candidates and measure only top-k.

## Candidate sources
- CPU/GPU tiling candidates: tvm_autotuning/results/tiling_candidates.jsonl
- FPGA sweep candidates: fpga_tiling_sweep/data/fpga_sweep_dataset.csv

## Predictor
- model:
- target:
- metrics (val/test): MAE/RMSE/MAPE/Spearman

## Selection results
- regret@5:
- regret@10:
- regret@20:

## Key discussion
- When does ranking matter more than absolute error?
- Under UART, why does compute optimization not reduce total latency much?
- What changes if we switch to PCIe/Ethernet?
