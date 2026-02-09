FPGA latency estimator equations (v0)

I/O
bytes_per_sec = baud / 10
T_io = bytes_total / bytes_per_sec

Compute
T_compute = cycles / f_clk

Cycles
FC:
cycles = IN + c0
DWConv1D:
cycles = C * (L-K+1) * K + c1

Total
T_total = T_io + T_compute + T_host

We report:
- bytes_in, bytes_out, baud, bytes_per_sec, T_io_ms
- cycles, f_clk, T_compute_ms
- T_total_ms

Why it is useful even if inaccurate
A cost model is valuable when it preserves ordering:
If model A is slower than B in reality, we want the estimator to rank it slower too.
That is enough for NAS and schedule selection loops.
