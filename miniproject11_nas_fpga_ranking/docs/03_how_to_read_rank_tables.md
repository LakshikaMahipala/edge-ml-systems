# How to read rank tables

Each candidate has multiple ranks:
- rank_reduced_train
- rank_snip
- rank_gradnorm
- rank_fpga

Interpretation:
- If rank_fpga is much worse than rank_reduced_train:
  candidate is "accurate but hardware expensive" (bad for FPGA target)
- If rank_fpga is good but rank_reduced_train is worse:
  candidate may be "hardware cheap" but needs training improvements
- If all ranks agree:
  candidate is robustly good under multiple signals
