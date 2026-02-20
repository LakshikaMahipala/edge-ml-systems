# Labels / training targets

Target y:
- latency_p50_ms or latency_p99_ms (choose one)
- measured end-to-end, not just kernel time

Optional multi-target:
- latency + energy proxy + memory BW

Recommendation:
Start with latency_p50_ms for stable training.
Use p99 later once measurement noise is controlled.

Store metadata with each label:
- device
- clock mode
- precision mode
- runtime (TensorRT/ORT/TVM)
