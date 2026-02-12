# Pruning intuition 

Goal
Remove parameters that contribute little to model output.

Why it helps
- smaller model size (storage)
- potentially fewer MACs (compute) if pruning is structured
- can reduce power and memory bandwidth

Two categories
1) Unstructured pruning: remove individual weights (sparsity)
2) Structured pruning: remove entire channels/neurons/filters (shape change)

Hidden truth
Unstructured sparsity often does NOT speed up inference unless:
- you have a sparse kernel/library
- sparsity is high enough (often >80–90%)
- memory access is efficient

Structured pruning is the usual path to real speedup on hardware.
