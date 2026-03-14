# Math tricks → FPGA mapping 

This doc answers: when does each math trick help on FPGA?

## Strassen (matrix multiply)
Helps when:
- very large GEMMs (N big)
- multipliers (DSPs) are the limiting resource
Hurts when:
- extra adds increase memory traffic
- recursion/control overhead dominates
FPGA takeaway:
- rarely the first lever; start with tiling + pipelining first.

## Winograd (3x3 conv)
Helps when:
- many 3x3 convs (classic CNNs)
- channels are high enough to amortize transforms
- weights are static (offline filter transform)
Hurts when:
- INT8/fixed-point scaling is hard (fractions)
- BRAM pressure increases due to transformed tiles
FPGA takeaway:
- can be strong for 3x3, but needs careful fixed-point design.

## FFT conv
Helps when:
- long 1D signals (vibration/audio/radar)
- large kernels
Hurts when:
- small kernels (3x3), overhead dominates
- buffering/control is expensive
FPGA takeaway:
- good for signal-processing style workloads, not typical CNN 3x3.

## Low-rank SVD (linear layers)
Helps when:
- linear layer is large and redundant
- you can choose rank aligned with unroll factors
Hurts when:
- rank not small enough
- intermediate vector bandwidth becomes bottleneck
FPGA takeaway:
- very practical for reducing DSP usage and memory, especially for large FC blocks.
