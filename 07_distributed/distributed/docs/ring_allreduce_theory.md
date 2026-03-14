Ring AllReduce 

Goal
Given N workers each holding a vector g_k (e.g., gradients), compute:
g_sum = sum_k g_k
and ensure every worker ends with g_sum (or g_avg = g_sum/N).

Why not a parameter server?
- Central bottleneck (bandwidth and latency).
- Ring spreads communication load evenly: each node talks only to neighbors.

Key decomposition
Ring allreduce is typically implemented as:
1) Reduce-Scatter
2) Allgather

Assume:
- N workers in a ring.
- Vector length L is divisible by N.
- Split each vector into N chunks: chunk 0..N-1 (each size L/N).

Phase 1: Reduce-Scatter (N-1 steps)
At step s, worker i sends a chunk to (i+1) and receives a chunk from (i-1).
Each worker accumulates (adds) the chunk it receives into its local chunk buffer.
After N-1 steps, each worker holds exactly 1 chunk of the global reduced sum.

Phase 2: Allgather (N-1 steps)
Now workers exchange chunks to assemble the full reduced vector on every worker.
After N-1 steps, every worker has all chunks → full sum vector.

Communication cost intuition
- Each worker sends/receives ~2*(N-1)/N * L elements total (for sum vector length L).
- No single node becomes the bottleneck; bandwidth is utilized uniformly.

What you should remember (hidden systems insight)
- Allreduce performance is dominated by bandwidth and topology.
- The ring is excellent when links are uniform and latency is not too high.
- Real systems use NCCL (GPU), RoCE/Infiniband, or specialized interconnects (NVLink).

In our repo today
We implement a CPU-only simulation:
- verifies correctness vs direct sum
- shows the two phases explicitly
