Zero-copy, Pinned Memory, and Why Copies Dominate 

Three memory types you must understand

1) Pageable host memory (normal numpy / malloc)
- OS can move these pages around.
- GPU DMA engine cannot safely DMA from it directly.
- CUDA driver typically stages through pinned buffers internally.
Result: H2D/D2H copies are slower + more variable.

2) Pinned (page-locked) host memory
- OS guarantees the pages won’t move.
- GPU can DMA directly (faster, lower jitter).
- Required for high-throughput streaming pipelines.
Tradeoff: too much pinned memory hurts OS performance.

3) Zero-copy (mapped host memory)
- GPU can access some host memory directly over PCIe (or SoC fabric on Jetson).
- Useful for small, latency-sensitive transfers where avoiding a copy helps.
- But bandwidth and latency depend heavily on platform; often slower than device DRAM for heavy compute.

What “zero-copy” really means
- It does NOT mean “free”.
- It means “no explicit memcpy”, but access still costs latency/bandwidth over interconnect.

What we measure
- H2D and D2H bandwidth and p50/p99 latency vs tensor size
- Compare pageable vs pinned buffers
- Use this to budget “copy time” in end-to-end inference

Why this matters for our plan
- Jetson + GPU: host<->device copies can dominate at batch=1
- FPGA over UART: transfers dominate even more, so we must compress/pack/stream
- This is why our repo includes a latency budget model (tools/latency_budget.py)
