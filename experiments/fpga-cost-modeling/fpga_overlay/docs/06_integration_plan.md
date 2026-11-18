# Integration plan (how this overlay fits the whole project)

Inputs:
- model compiler/exporter produces layer descriptors (JSON)
- runtime sends descriptor + tensor pointers to FPGA overlay

Outputs:
- FPGA runs layer, writes output tensor
- host orchestrates next layer

Later steps:
- implement a small descriptor parser + controller FSM in RTL
- implement conv core + BN/ReLU post-op
- validate with 1 layer first, then chain layers
