Pipeline + Queueing Model 

Definitions
We model an inference system as a pipeline of stages:

Stage A: Preprocess
Stage B: Inference (GPU/TRT/FPGA/etc.)
Stage C: Postprocess + packaging

Each stage has a service time:
t_pre, t_inf, t_post (milliseconds per item)

Key outputs
1) Latency per item (single-item end-to-end)
T_latency ≈ t_pre + t_inf + t_post
(+ copies if relevant)

2) Throughput (steady-state)
For a strictly sequential single-worker system:
TPS ≈ 1 / (t_pre + t_inf + t_post)

For a pipelined system with separate workers per stage:
TPS ≈ 1 / max(t_pre, t_inf, t_post)
Because the slowest stage dominates.

This is the core pipeline insight:
- latency adds
- throughput bottlenecks

Queueing reality (why p99 grows)
If arrivals come faster than service capacity, queues build up.
Even if average arrival rate is slightly below capacity, variability can cause queueing spikes.
That creates tail latency.

Utilization
For a bottleneck stage with service rate μ = 1/t_bottleneck and arrival rate λ:
ρ = λ / μ
As ρ approaches 1, queueing delay can explode.

Practical rule
- If you want stable p99 latency, do not run near ρ = 1.
- You need headroom (e.g., keep ρ < ~0.7–0.8 for real-time systems).

Why this matters for our repo
- inference_bench gives stage breakdown (pre/inf/post).
- transfer_bench gives copy cost.
- video_demo is a live pipeline that will show bottlenecks.
- fpga offload only helps if inference is the bottleneck and copies don’t dominate.

We will use scripts:
- pipeline_budget.py : compute bottleneck TPS and estimated latency
- pipeline_sim.py    : simulate a queue with arrivals and measure p50/p99

Important: our repo will always report:
- stage times
- end-to-end latency
- bottleneck analysis
