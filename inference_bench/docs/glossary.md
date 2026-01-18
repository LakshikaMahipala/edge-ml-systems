Glossary (ML Hardware & Systems)

Latency:
Time to process one item (e.g., one image) end-to-end. Often reported as p50/p99.

Throughput:
Items processed per second (e.g., images/sec). Batch size often increases throughput.

Warmup:
Initial iterations that are not measured because caches, JIT, memory allocation, and runtime setup can distort timings.

p50 (median):
50% of measurements are faster than this value. Represents “typical” latency.

p99 (tail latency):
99% of measurements are faster than this value. Represents “worst-case” latency that matters in real systems.

Top-1 Accuracy:
Prediction is correct if the highest-score class matches the ground truth.

Top-5 Accuracy:
Prediction is correct if the ground truth appears in the top 5 predicted classes.

End-to-end latency:
Total time including preprocessing + inference + postprocessing (and transfers, if any).

Kernel time:
Time spent inside the core compute operation only (e.g., GPU kernel). This is NOT end-to-end.
