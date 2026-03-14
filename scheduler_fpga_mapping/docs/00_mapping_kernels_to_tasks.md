# Mapping FPGA kernels into scheduling models 

A scheduler needs, for each task:
- execution time on each device (CPU/GPU/FPGA)
- communication cost between tasks if placed on different devices

We already built:
- FPGA kernels (BNN, INT8 MLP/conv-lite)
- FPGA proxy cost models
- transfer budget models (PCIe vs network vs UART)

Today we connect them:
- define a latency registry (measured or proxy)
- generate a pipeline DAG (pre → infer → post etc.)
- run HEFT / OPT and compare schedules

Key insight:
Even if FPGA kernel compute is fast,
IO and dependency placement can eliminate speedup.
