Metrics Table (single source of truth)

Rules
- Always report p50 and p99 latency (ms)
- Always state: device, batch size, input size, and what "end-to-end" includes
- If accuracy is reported, specify dataset subset used

Table

| Project | Model | Device | Batch | Input | p50 (ms) | p99 (ms) | Top-1 | Top-5 | Notes |
|--------|-------|--------|-------|-------|----------|----------|-------|-------|------|
| inference_bench | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Day 2 will fill this |
| inference_bench | resnet18 | cpu | 1 | 1x3x224x224 | TBD | TBD | TBD | TBD | Week2 Day6 Path A: Python preprocess |
| inference_bench | resnet18 | cpu | 1 | 1x3x224x224 | TBD | TBD | TBD | TBD | Week2 Day6 Path B: C++ preprocess (pybind11 + SIMD) |
| inference_bench | file_io_read | cpu | - | file_path=TBD | p50=TBD | p99=TBD | mean=TBD | throughput=TBD | Week3 Day1 file I/O benchmark; run later |



