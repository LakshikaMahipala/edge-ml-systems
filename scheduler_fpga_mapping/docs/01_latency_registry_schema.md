# Latency registry schema

We define a registry of kernel times per device.

Example keys:
- conv3_16x16_32
- conv5_24x24_32
- bnn_dot_256
- mlp_256x128

Each entry stores:
- device: cpu/gpu/fpga
- latency_p50_ms (or proxy)
- latency_p99_ms (optional)
- measurement metadata (clock, link, precision)

This registry lets scheduling experiments be reproducible and updatable.
