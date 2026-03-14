Micro-NPU Platform Spec 

This folder defines a minimal platform:
- operator set
- tensor/quant rules
- buffer map
- command stream
- compiler interface
- timing model hooks

It connects directly to:
- fpga kernels (int8 FC, dwconv1d)
- host async driver (UART)
- performance_notebook (I/O-first evaluation)
