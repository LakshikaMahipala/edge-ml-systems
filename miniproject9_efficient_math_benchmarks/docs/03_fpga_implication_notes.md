# FPGA implication notes (Mini-project 9)

## The rule
A math trick helps only if it maps to a datapath that improves:
- compute cycles and/or
- memory traffic and/or
- power
under the real IO regime.

## Strassen
- reduces multiplications (DSP pressure)
- increases additions and buffering
- likely irrelevant until GEMM is huge and DSP-limited

## Winograd (3x3 conv)
- strong candidate for FPGA CNN kernels
- must implement fixed-point transforms carefully (fractions)
- BRAM buffering per tile is a first-order design constraint
- best when channels are large enough to amortize transform overhead

## FFT conv
- strongest for long 1D signals + large kernels
- hardware cost is dominated by buffering + butterfly network control
- generally not worth it for standard 3x3 CNN inference

## Low-rank SVD
- practical and common: replace one big FC with two smaller FCs
- choose rank r aligned to unroll factor for clean datapaths
- intermediate vector size r creates bandwidth and buffering needs

## What to measure later (must-have)
- cycles from RTL counters (compute vs transform)
- BRAM usage at synthesis
- end-to-end latency with realistic IO (PCIe/Ethernet vs UART)
- stability: p99 latency under streaming load
