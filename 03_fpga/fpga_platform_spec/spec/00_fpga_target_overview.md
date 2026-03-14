FPGA Target Overview 

Goal
Define a stable FPGA “backend contract” for our Micro-NPU platform:
- what ops exist
- what inputs/outputs look like over transport
- what buffer map is assumed
- what timing model fields are reported

FPGA kernels available
- INT8_FC (v0 combinational, v1 pipelined)
- INT8_DWCONV1D_K3 (v0 sequential)

Transport today
- UART framed packets with CRC (host_tools + host_tools_async)

Transport later (same logical API)
- PCIe/M.2/Ethernet with identical buffer/command semantics

Principle
We do NOT rewrite host code per kernel.
We enforce a stable interface so kernels can evolve independently.
