# Accuracy failure modes (BNN)

BNNs often fail because:
- limited precision cannot represent small changes
- gradients are noisy due to STE approximation
- activation distributions saturate

Common fixes (later in Week 12/13):
- use BatchNorm aggressively
- use wider networks
- use scaling factors (α) per layer (XNOR-Net style)
- keep first/last layers in FP16/INT8
- use better binarization (learned thresholds)
