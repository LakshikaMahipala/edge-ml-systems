# Benchmark protocol

We measure:
- preprocess time (if any)
- inference time (forward pass)
- end-to-end time
Metrics:
- latency p50 / p99 (ms) using warmup + iterations
- accuracy: placeholder until real dataset is used (synthetic gives random)

Later:
Replace synthetic dataset with CIFAR10 and report real accuracy.

Important:
Timing must be isolated:
- CPU: use time.perf_counter + torch.no_grad
- GPU (later): cuda synchronize
