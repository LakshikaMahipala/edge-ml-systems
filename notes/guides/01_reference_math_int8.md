Reference INT8 math (used for both micro and FPGA)

We compute:
acc32 = sum(x_int8 * w_int8) + bias_int32

Then requant:
y_int8 = sat8( round_shift(acc32, SHIFT) )

ReLU for int8 (simplified):
y = max(y, 0)

Notes
- This is a simplified quantization model for learning + RTL verification.
- Real deployments use scale/zero-point and multiplier+shift.
- Our rule is bit-exact reproducibility under this simplified spec.
