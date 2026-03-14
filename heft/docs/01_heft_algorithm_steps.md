# HEFT algorithm steps (exact)

Inputs:
- DAG tasks with edges (dependencies)
- computation cost w(t, p): time for task t on processor p
- communication cost c(i, j): cost on edge i→j if on different processors

Step A: Compute upward rank for each task:
rank_u(t) = avg_w(t) + max_{t→s} ( avg_c(t,s) + rank_u(s) )
(exit tasks: rank_u = avg_w)

Step B: Sort tasks in descending rank_u.

Step C: For each task t in order:
- for each processor p:
  compute earliest start time EST(t,p) considering:
  - dependencies finish time + comm
  - earliest idle slot on p
  then EFT(t,p) = EST + w(t,p)
- assign t to processor with minimum EFT.

Outputs:
- mapping task→processor
- start/end times
