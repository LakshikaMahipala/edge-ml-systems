# Tiling + dataflow (how we fit in BRAM)

We tile because full tensors don't fit on-chip.

Common tiling knobs:
- Tm: output channels per tile
- Tn: input channels per tile
- Tr,Tc: spatial tile (rows/cols)

Dataflow options:
1) Weight-stationary: keep weights in BRAM while streaming activations
2) Output-stationary: keep partial sums on-chip, accumulate across input channels
3) Row-stationary: balanced reuse

For Conv/BN/ReLU overlay:
Output-stationary is common:
- keep psum tile in PBuf
- loop over input channel tiles, accumulate
- then BN+ReLU and write out

Design goal:
Minimize external memory traffic.
