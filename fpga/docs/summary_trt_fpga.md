TensorRT + FPGA Bring-up

TensorRT track (completed scaffolding)
- Export PyTorch -> ONNX
- Build TensorRT engines:
  - FP32 baseline
  - FP16 build option
  - INT8 PTQ build with calibration cache
- Benchmark command runner (trtexec) with warmup/iters
- Parse logs to JSON containing throughput + latency mean/p50/p99
- FP32 vs INT8 drift proxy (top-1 agreement + cosine logits)

FPGA track (completed foundations)
- hello_uart_echo:
  - UART RX/TX and echo top
  - Self-checking testbench
  - iverilog simulation script
- fixed_point_lib:
  - Q-format primitives (add/mul/saturate/scale_shift)
  - Unit tests + sim script

Mini-project 3 (report scaffold)
- TRT vs ORT vs PyTorch comparison report template
- Latency budget model + tool to compute speedup bounds
- FPGA appendix: where FPGA sits in the pipeline (compute vs I/O)

Next week (Week 6)
- Jetson stack study (deployment thinking)
- FPGA ML kernel v0: INT8 linear layer / small MLP in RTL or HLS
- Host->FPGA protocol and end-to-end latency measurement (UART ok)
