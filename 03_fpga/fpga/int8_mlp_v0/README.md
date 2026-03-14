INT8 MLP v0 

Goal
- Implement an INT8 fully-connected layer:
  y = requant( sum_i x[i]*w[j,i] + b[j] )

Key design choices
- x: int8
- w: int8
- accum: int32
- requant: right shift + rounding + saturate -> int8

This v0 uses fixed compile-time dimensions:
- IN = 8
- OUT = 4
Weights/bias are hardcoded constants in RTL for simulation correctness.

Run (later)
- Install iverilog
- chmod +x scripts/sim_iverilog.sh
- ./scripts/sim_iverilog.sh
Expected:
- PASS
