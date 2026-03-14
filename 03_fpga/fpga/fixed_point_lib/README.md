Fixed-Point Library 

Goal
- Implement and verify fixed-point primitives used in INT8 inference kernels:
  add, mul, saturate, scaling shifts.

Q-format used
- Signed two’s complement integers with F fractional bits.
- Real value = stored_int / (2^F)

Modules
- fxp_add: saturated add
- fxp_mul: multiply with downshift (>> F) + rounding + saturation
- fxp_saturate: generic clip to N-bit signed
- fxp_scale_shift: apply scale by shifting (useful for requantization)

How to simulate (later)
- Install iverilog
- chmod +x scripts/sim_all_iverilog.sh
- ./scripts/sim_all_iverilog.sh
Expected:
- All tests PASS
