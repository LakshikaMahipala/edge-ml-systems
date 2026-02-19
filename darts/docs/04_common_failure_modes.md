# DARTS failure modes (what papers don't emphasize enough)

1) Skip-connection dominance
- skip ops are cheap, yield strong gradients early
- α shifts toward skips and collapses search

2) "Cheap op bias"
- because mixed ops compute all candidates, gradients may favor low-cost ops

3) Weight coupling / unfair comparison
- all ops share the same training process and compete through gradients
- some ops need different learning rates or warmup to look good

4) Validation leakage (bad splitting)
- if train/val split is weak, α optimizes the wrong objective

5) Discretization gap
- best mixed network ≠ best discrete network after argmax selection

Mitigations (we will implement later):
- regularize α (entropy, sparsity penalties)
- drop-path / path dropout
- restrict or delay skip connections
- add latency regularizer (Week 14 Day 4)
