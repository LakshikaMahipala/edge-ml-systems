I/O overhead is first-class (why FPGA must include it)

FPGA is not “just faster compute”.
FPGA requires moving tensors between host and FPGA.

End-to-end time must include:
- host->fpga transfer
- fpga compute
- fpga->host transfer
- protocol overhead (UART framing, CRC, buffering)
- host scheduling overhead (async driver, queues)

Common mistake (what we avoid)
Claiming speedup using only compute time:
Speedup_fake = T_cpu_compute / T_fpga_compute

Correct evaluation:
Speedup_real = (T_pre + T_cpu_inf + T_post + T_io_cpu)
               -----------------------------------------
               (T_pre + T_fpga_inf + T_post + T_io_fpga)

UART reality
UART bandwidth is low, so for many models:
T_io_fpga dominates → FPGA compute speed is irrelevant.
That’s fine. We measure it honestly.

When FPGA becomes meaningful
- faster interface (PCIe, M.2, Ethernet)
- larger batch sizes / more compute per transferred byte
- better overlap (double buffering + streaming)
