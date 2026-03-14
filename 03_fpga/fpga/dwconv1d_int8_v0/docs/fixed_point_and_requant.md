Fixed-point + requantization

We compute int32 accumulation:
acc = sum(int8 * int8) -> int32

Then we map back to int8:
y = sat8( round_shift(acc + bias, SHIFT) )

SHIFT is a simplified proxy for scale.
In real quantized inference:
- each layer has scale/zero-point
- requant uses multiplier + shift (often per-channel)

For our FPGA correctness kernel:
- we use SHIFT only to keep it simple
- later we will upgrade to multiplier+shift if needed
