Pipeline overview

Inputs
- Relay IR text file: tvm_relay_intro/outputs/relay_*.txt

Steps
A) relay_op_counter.py
- counts common ops (conv2d, dense, relu, add, …)
Output: histogram printed to terminal (later: save to results)

B) relay_pattern_rewrite_toy.py
- produces a rewritten Relay text file (toy demonstration)
Output: rewritten Relay text

C) relay_to_cost.py (this mini-project)
- reads Relay text OR a relay snippet
- maps a supported subset into our FPGA op graph JSON:
  - nn.dense -> INT8_FC (placeholder mapping)
  - depthwise conv -> INT8_DWCONV1D_K3 (placeholder mapping)
- then calls fpga_cost_model estimator

D) fpga_latency_estimator_v0.py
- returns T_io, T_compute, T_total

Outputs
- results JSON breakdown (cost model)
- markdown summary (expected results template)

Note
Today the mapping is a “demo mapping”, not full correctness.
The goal is to show the architecture of a HW-aware compilation pipeline.
