Comparison report template (fill after local runs)

Block
- DWConv1D K=3 + ReLU (C=4, L=16)

Backends compared
A) Micro runtime estimate (MAC/cycle proxy)
B) FPGA kernel estimate (cycles + UART I/O)
C) Measured end-to-end (later)

A) Micro runtime (estimated)
- MACs (DWConv): TBD
- assumed alpha cycles/MAC: TBD
- estimated cycles: TBD
- memory footprint: input bytes, weights bytes, output bytes

B) FPGA kernel (estimated)
- compute cycles: TBD
- f_clk assumed: TBD -> T_compute
- UART bytes in/out: TBD
- baud: TBD -> T_uart_io
- predicted end-to-end: TBD

C) Measured (later)
- p50/p99 end-to-end latency
- timeout rate
- CRC errors

Conclusion (expected pattern under UART)
- UART dominates. FPGA compute speed does not change end-to-end much.
- This is not failure; it is the correct system conclusion.
- Next step: faster interface or more compute per transferred byte.
