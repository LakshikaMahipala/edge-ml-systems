# Latency prediction with GNN 

Goal:
Predict end-to-end inference latency of a neural network on a target device
without benchmarking every candidate.

Why:
NAS/co-design needs to evaluate thousands of architectures.
Real measurement is too slow.

Solution ladder:
- FLOPs proxy (cheap, inaccurate)
- layerwise sum model (better)
- learned predictor (best scaling)
- GNN predictor (best generalization for graph structure)

We focus on GNN because models are computation graphs, not sequences.
