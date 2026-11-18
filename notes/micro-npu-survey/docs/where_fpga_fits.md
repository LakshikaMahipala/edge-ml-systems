Where FPGA fits (in the Micro-NPU landscape)

FPGA is not a “default choice”.
FPGA is chosen when you need something that fixed NPUs cannot do.

FPGA strengths
1) Custom dataflow
- You can build exactly the loop nest you want (tiling/unrolling/pipelining).

2) Custom precision
- INT8/INT4/BNN, mixed precision, saturating arithmetic, custom scaling.

3) Deterministic latency (streaming)
- Pipelines can provide stable latency for sensor streams.

4) Custom operators
- If you need an operator not supported by NPU compilers, FPGA can implement it.

FPGA weaknesses
1) I/O bottleneck can erase compute gains
- UART makes FPGA compute irrelevant for many workloads.
- PCIe/M.2/Ethernet changes the equation.

2) Engineering cost
- RTL/HLS + verification + toolchain overhead.

3) Model portability
- You must define your own “compiler interface” or integrate into existing flows.

Practical FPGA role in our repo
- Start with a small kernel (INT8 FC / depthwise block) + correct I/O.
- Measure end-to-end honestly (compute + I/O).
- Use the cost model + Amdahl tools to decide if FPGA is worth scaling.

The professional takeaway
FPGA is best thought of as:
a programmable Micro-NPU you can shape to your workload
— but only if the system I/O supports it.
