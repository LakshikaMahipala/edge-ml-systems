# Consolidation

Completed modules
- micro_npu_survey: taxonomy + where FPGA fits
- tinyml_micro: TFLite Micro/CMSIS-NN skeleton + MAC proxy estimator
- fpga/dwconv1d_int8_v0: INT8 depthwise conv kernel + TB
- micro_npu_platform: operator set + buffer map + command stream spec
- fpga_platform_spec: FPGA backend contract + UART transport spec
- miniproject6_mobilenet_block: reference Python block + MAC estimator + comparison template

Key platform insight (Week 8)
- “Kernel speed” is not the product.
- The product is the platform contract + I/O + measurement discipline.
- Under UART, FPGA compute is almost always hidden by I/O.

Next week (Week 9)
- TVM/Relay + FPGA cost model hook
