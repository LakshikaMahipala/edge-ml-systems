# Feature spec (node + edge features)

## Node features (per operator node)
Minimal set (works surprisingly well):
- op_type (one-hot / embedding): conv, dwconv, matmul, add, relu, bn, etc.
- input shape: (Cin, Hin, Win) or (B, D)
- output shape: (Cout, Hout, Wout)
- kernel size (kH,kW) if conv
- stride (sH,sW) if conv
- groups if grouped/depthwise conv
- padding
- dilation
- activation dtype (fp32/fp16/int8)
- weight dtype (fp32/fp16/int8)
- MACs estimate (scalar)
- parameter count (scalar)

Hardware-awareness features (optional but strong):
- estimated bytes_read, bytes_write
- arithmetic intensity = MACs / bytes_moved
- cache fit flag (rough): tensor_bytes < L2_size

## Edge features (tensor-flow edges)
- tensor size in bytes (activation bytes)
- layout hint (NCHW vs NHWC) if known
- quantization scale/zero-point availability (if int8)
- dependency type (normal, residual add, concat)

Global graph features
- batch size
- target device ID (embedding) OR device specs (SM count, clock, bandwidth)
- runtime flags: fusion enabled, precision mode

Why these features:
Latency is shaped by:
- compute (MACs)
- memory movement (bytes)
- parallelization/fusion patterns (graph structure)
GNN sees structure; features tell it magnitudes.
