# Few-shot training protocol

Problem:
We may only have a small number of real device measurements.

Solution approach:
1) Pretrain a predictor on synthetic/proxy labels:
   - op-level latency tables
   - simulator estimates
2) Fine-tune with few real measurements on target device

Practical recipe:
- collect 50–200 measured graphs if possible
- start with 10–20 if not possible (few-shot)
- freeze part of the model (message passing layers) and tune only head
- use strong regularization + early stopping

If extremely low data:
- use pairwise ranking loss (A faster than B) instead of regression
