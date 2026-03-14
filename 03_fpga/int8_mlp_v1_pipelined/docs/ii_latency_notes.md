Initiation Interval (II) and Latency (FPGA pipelining)

We compute:
y[j] = sum_i x[i]*w[j,i] + b[j]

In v0 (combinational):
- Latency: 0 cycles (combinational), but timing is unrealistic for large IN
- Throughput: 1 vector/cycle in theory, but impossible once IN grows

In v1 (pipelined MAC):
- We process 1 input element per cycle.
- Maintain OUT accumulators in parallel.
- After IN cycles, all OUT sums are ready and requantized.

Key numbers
- cycles_per_vector = IN
- II_vector (accept rate) = IN (in this simple single-buffer design)
- latency_first_output ≈ IN cycles (+ requant)

Why this is “real”
This architecture scales: you trade hardware (OUT accumulators) for throughput.
Later improvements:
- double buffering to accept next vector earlier
- deeper pipelining, partial unrolling, tiling
