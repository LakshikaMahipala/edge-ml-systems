How to use DLA in TensorRT (practical notes)

Key idea
TensorRT can target DLA for supported layers and run the remaining layers on GPU. This is configured at build time. (DLA supports only a subset of layers.)  [TensorRT docs]

Two modes
1) DLA + GPU fallback (recommended for first runs)
- Build engine targeting DLA but allow GPU fallback for unsupported layers.
- This keeps the workflow robust.

2) DLA-only (strict validation)
- Build engine that must run entirely on DLA.
- Used to test whether a model is DLA-compatible.

What to record in reports
- Did the engine build succeed?
- Which layers ran on DLA vs GPU? (from build logs)
- Latency p50/p99
- Precision mode (FP16/INT8)
