TensorRT Bench Results Schema

File: results/*.json

Top-level keys

1) meta
- model: string (e.g., resnet18)
- precision: string (fp32/fp16/int8)
- batch: int
- device: string (gpu name later, placeholder now)

2) metrics
- throughput_qps: float
- latency_mean_ms: float
- latency_p50_ms: float
- latency_p99_ms: float

3) source_log
- path to the trtexec log used to compute these metrics
