# Latency budget model (end-to-end)

End-to-end latency is not "kernel time".
It is:

T_total =
  T_preprocess +
  T_host_to_device +
  T_device_compute +
  T_device_to_host +
  T_postprocess +
  T_queueing/jitter

Key hidden truth:
If transfer dominates, faster kernels will not improve end-to-end latency much (Amdahl).

Transfer cost proxy:
T_copy ≈ payload_bytes / effective_bandwidth + fixed_overhead

Overhead includes:
- driver overhead
- DMA setup
- RPC framing (network-attached)
- serialization/deserialization
