FPGA timing model (v0)

We split latency into:
T_total = T_uart_io + T_fpga_compute + T_host_overhead

1) UART I/O time
T_uart_io ≈ bytes_total / (baud_rate_bytes_per_sec)

baud_rate_bytes_per_sec ~ baud/10 (start + stop bits)
Example: 115200 baud -> ~11520 bytes/s

So 1000 bytes round-trip ≈ 86.8 ms (dominant).

2) FPGA compute time (cycles)
T_compute = cycles / f_clk

INT8_FC v1:
cycles ≈ IN (+ small overhead)
DWCONV1D v0:
cycles ≈ C*(L-2)*K

3) Host overhead
Includes:
- Python scheduling
- serial driver buffering
- queueing delays under load

What we report
For each run we store:
- bytes_in, bytes_out
- assumed baud and estimated T_uart_io
- cycles_est and assumed f_clk -> T_compute_est
- measured end-to-end latency (later)

This lets us explain performance honestly:
If UART dominates, FPGA compute improvements won’t help until interface changes.
