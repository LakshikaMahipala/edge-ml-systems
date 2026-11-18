Results Policy (what we commit vs what we don’t)

We commit (✅)
- Small JSON summaries (latency p50/p99, throughput, metadata)
- Markdown result tables in docs/
- Small plots (png) if they are under ~1–2 MB
- Logs only if small and essential

We do NOT commit (❌)
- Large binary TensorRT engines (.plan) — they are device-specific and huge
- Large datasets
- Full video outputs unless tiny demo clips

Folder policy
- Each module has results/ with a .gitignore that ignores artifacts by default.
- We commit only the curated summaries.

Naming convention
- <model>_<backend>_<precision>_b<batch>_<device>.json
Examples:
- resnet18_trt_fp16_b1_jetson.json
- mobilenetv3_ort_cpu_b1_laptop.json

Minimum metrics for any run
- p50 latency (ms)
- p99 latency (ms)
- mean latency (ms)
- throughput (qps) if applicable
- warmup iters, measured iters
- device + software versions
