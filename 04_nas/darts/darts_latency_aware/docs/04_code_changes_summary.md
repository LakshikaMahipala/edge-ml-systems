# Code changes summary

We implement:
- latency_table.py: base costs for ops
- latency_regularizer.py: compute expected latency from MixedOps
- trainer_latency_aware.py: same as DARTS trainer, but alpha-step uses:
  loss_alpha = L_val + λ * latency_regularizer

We also export:
- genotype + total expected latency score
