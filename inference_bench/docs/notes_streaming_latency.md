Streaming Latency Notes 

Key idea: latency = queue wait + service time
- queue wait: time spent waiting in queue before processing
- service time: time spent in preprocess + inference + postprocess
- end-to-end: total time from enqueue to completion

Why p99 matters
Real-time systems fail on tail events.
Even if p50 is good, occasional spikes can break SLAs.

Closed-loop vs open-loop
Closed-loop (backpressure):
- producer blocks when queue is full
- protects p99 latency
- throughput limited by consumer capacity

Open-loop (fixed-rate input):
- simulates camera FPS
- if input rate > service rate, queue grows
- latency grows; eventually either drops or unbounded delay

Drop policy
If drop_when_full=True:
- system chooses to drop frames instead of increasing latency
- common in real-time video systems where “freshness” is more important than completeness

What we implemented
- per-item timing envelope in pipeline
- summaries: p50/p99 for queue wait, service time, end-to-end latency
