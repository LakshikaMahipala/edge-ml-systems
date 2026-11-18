# Notes

Study notes and decision guides written while building the projects. Useful as a reading path for students moving from embedded systems into ML systems.

## Deployment decision guides — [`guides/`](guides)
- **Which backend?** [TensorRT vs TVM vs FPGA](guides/trt_vs_tvm_vs_fpga_decision_matrix.md) · [decision matrix](guides/decision_matrix.md) · [which optimization helps which layer](guides/summary_which_trick_helps_which_layer.md) · [when a binary NN on FPGA makes sense](guides/when_fpga_bnn_makes_sense.md) · [compression for FPGA](guides/decision_guide_compression_fpga.md)
- **NVIDIA Jetson:** [stack overview](guides/jetson_stack_overview.md) · [deployment playbook](guides/jetson_deployment_playbook.md) · [ONNX Runtime GPU](guides/jetson_onnxruntime_gpu_notes.md) · [version matrix](guides/jetson_version_matrix.md) · [DLA basics](guides/dla_basics_and_selection_logic.md) · [DLA debug checklist](guides/dla_debug_checklist.md)
- **End-to-end performance:** [bottleneck examples](guides/end_to_end_bottleneck_examples.md) · [latency / throughput / energy](guides/03_latency_throughput_energy_discussion.md) · [proxy reliability](guides/proxy_reliability_compute_budget.md) · [results policy](guides/results_policy.md)
- **INT8 maths:** [reference INT8 arithmetic](guides/01_reference_math_int8.md)

## FPGA — [`fpga/`](fpga), [`fpga-platform-spec/`](fpga-platform-spec), [`lutnet/`](lutnet)
Vivado setup and toolchain choice, latency budget model, FPGA platform specification (op interfaces, buffer map, UART transport, tiling, timing model), and notes on LUTNet (computing directly in FPGA lookup tables).

## Micro-NPU — [`micro-npu-survey/`](micro-npu-survey), [`micro-npu-platform/`](micro-npu-platform)
A survey of microcontroller-class neural accelerators and a design sketch for a small NPU platform.

## Learning log — [`learning-path.md`](learning-path.md)
The original day-by-day plan that produced this repository.
