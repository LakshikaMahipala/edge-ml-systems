Profiling Notes 

Why profiling exists
Timing tells you “how fast.” Profiling tells you “where time goes.”
In ML systems work, guessing is unacceptable; you must measure bottlenecks.

Timing vs Profiling
- Timing (Timer): gives p50/p99 latency for a whole function or pipeline.
- Profiling (torch.profiler): breaks time down by operators and call sites.

Common pitfalls
- Measuring without warmup (first runs include setup noise)
- Printing/logging inside timed loops
- Reporting only average latency (tail latency matters)
- Confusing kernel time (compute only) with end-to-end latency (real system)

What to capture in every report
- Device + batch size + input size
- Warmup iterations + measured iterations
- p50 and p99 end-to-end
- Component breakdown: preprocess, inference, postprocess
