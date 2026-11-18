Regimes + numerical precision

Winograd works best for:
- 3x3 convolutions
- larger channel counts (transform overhead amortized)
- FP32/FP16 contexts with stable transforms

Where it can be problematic:
- INT8: transform introduces fractions; quantization error can grow
- very small channel counts: overhead dominates
- boundary handling: padding complicates tiling

Engineering reality
Many libraries enable Winograd automatically when it helps (cuDNN does this for some cases).
But for hardware co-design, we must model:
- arithmetic count changes
- memory footprint changes
- numerical stability constraints
