# FPGA overlay intuition 

Two ways to run NN layers on FPGA:

A) Fixed-function RTL per model
- You generate hardware specifically for one network
- Best efficiency
- Slow iteration, hard to reuse

B) Overlay accelerator (config-driven)
- You build a reusable datapath (e.g., Conv/BN/ReLU engine)
- Each layer is described by a "layer descriptor" (like an instruction)
- Same hardware runs many models

Overlay is basically:
"FPGA as a programmable accelerator" with a small custom ISA.

Why overlays matter:
- faster development
- easier deployment
- software stack can target one stable interface
