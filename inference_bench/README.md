# Inference Benchmark Foundations 

### Latency vs Throughput
- Latency: time to complete one unit of work (often reported as p50/p99).
- Throughput: work completed per second (e.g., images/sec). Batching often increases throughput.

### Top-1 vs Top-5
- Top-1: correct if the highest-score predicted class equals the ground-truth label.
- Top-5: correct if the ground-truth label appears in the top 5 predictions.

