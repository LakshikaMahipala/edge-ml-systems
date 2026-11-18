# Cost terms and assumptions (FPGA cost model v1)

Total latency (ms) =
  IO latency + compute latency + transform overhead latency

We model:

## IO latency
- UART regime (today): T_io = bytes_total / (baud/10)
- Future PCIe/Ethernet regime: replace with bandwidth GB/s model

## Compute latency
Compute cycles ≈ MACs / parallelism * II
parallelism is a function of UNROLL factors.

## Transform overhead (math tricks)
- Winograd: input transform + output transform per tile; filter transform offline
- FFT: FFT(x) + IFFT(y) per signal; FFT(h) offline
- Low-rank: two matmuls + intermediate vector write/read (if needed)

## Resource proxies
- DSP proxy ~ unroll lanes
- BRAM proxy ~ tile buffers + transform buffers

All terms are first-order proxies, intended for ranking and budgeting,
not final silicon-level timing.
