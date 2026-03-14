# Throughput vs makespan (why scheduling matters)

Makespan = completion time of one DAG execution.

Throughput (steady state) depends on:
- pipeline parallelism
- bottleneck stage
- overlap between devices

In our simplified scheduler evidence:
- we compare makespan for different device placements
- lower makespan implies higher throughput for single-stream execution
- later we extend to multi-stream pipeline scheduling

Even without full throughput simulation,
makespan comparisons still show scheduling impact.
