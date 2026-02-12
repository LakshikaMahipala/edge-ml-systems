# INT8 quantization basics

Quantize a float tensor x to int8 q:
q = clamp(round(x / s) + z, qmin, qmax)

Dequantize:
x_hat = (q - z) * s

Where:
- s = scale (float)
- z = zero-point (int)
- qmin,qmax typically [-128,127] or [0,255] depending on signed/unsigned

Two common modes:
- per-tensor: one scale for entire tensor
- per-channel: one scale per output channel (much better for weights)
