FPGA Platform Spec 

This spec defines the FPGA backend contract for the Micro-NPU platform.

Connects to
- micro_npu_platform/spec (command stream, buffers)
- fpga kernels (int8 FC, dwconv1d)
- host_tools_async (req_id request/response matching)
- performance_notebook (I/O-first evaluation)

Transport today
- UART

Transport later
- PCIe/Ethernet with identical logical semantics
