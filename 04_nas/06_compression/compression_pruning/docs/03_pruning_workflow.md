# Pruning workflow

1) Train baseline model (FP32 or QAT model)
2) Prune (unstructured or structured)
3) Optional: fine-tune after pruning (usually needed)
4) Evaluate:
   - accuracy
   - sparsity (percent zeros)
   - model size
   - inference time (or cost proxy)
5) Decide:
   - if structured pruning: deploy smaller dense model
   - if unstructured: only claim speedup if sparse backend exists
