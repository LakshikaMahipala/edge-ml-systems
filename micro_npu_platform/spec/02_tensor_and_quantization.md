Tensor representation and quantization (v0)

Tensor layout
- 1D: [C][L] stored as contiguous C-major:
  index = c*L + t
- FC input: [IN]
- FC output: [OUT]

Data types
- activations: int8
- weights: int8
- bias: int32 (or int32-like pre-shifted form)
- accumulation: int32

Quantization model (simplified v0)
We use a SHIFT-based requant:
y_int8 = sat8( round_shift(acc_int32 + bias_int32, SHIFT) )

Notes
- Real quantization uses scale and zero-point (affine).
- Real requant uses multiplier + shift (often per-channel).
- SHIFT-only is a correctness scaffold; we upgrade later.

Saturation
After requant, clamp to int8 range [-128, 127].

Determinism requirement
Given the same input bytes and weights, output must match bit-exactly.
