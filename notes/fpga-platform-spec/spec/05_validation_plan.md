FPGA backend validation plan (v0)

Correctness
- For each op, compare FPGA outputs against a CPU reference implementation:
  - bit-exact for our SHIFT-based quantization

Tests
1) FC correctness
- random int8 vectors, deterministic weights
- compare y_fpga vs y_cpu

2) DWConv1D correctness
- random int8 inputs
- compare full output tensor

Transport reliability
- ping/pong stability test (1000 pings)
- measure timeout rate, CRC error count
- stress test host async queue with many requests

Performance reporting
- report p50/p99 latency
- report breakdown:
  - estimated UART time
  - estimated compute time
  - residual overhead

Success criteria (week 8)
- deterministic correct outputs for at least 20 random tests
- stable request/response with <1% timeouts under moderate load
