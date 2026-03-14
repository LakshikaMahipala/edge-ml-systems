# UART vs PCIe regimes

UART (115200 baud):
- ~11.5 KB/s
- IO dominates for anything > a few KB

PCIe Gen3 x4 (order-of-magnitude):
- hundreds of MB/s to a few GB/s effective depending on stack
- compute becomes visible

Practical implication:
Any FPGA “speedup claims” must specify the IO regime.
Under UART, math tricks won't change end-to-end latency meaningfully.
Under PCIe/Ethernet, compute optimizations (Winograd/SVD) can dominate.
