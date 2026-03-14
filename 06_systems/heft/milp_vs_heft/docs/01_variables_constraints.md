# Variables and constraints (detail)

1) Assignment:
Σ_p x(i,p) = 1  for all tasks i

2) Task finish time:
f_i = s_i + Σ_p x(i,p)*w(i,p)

3) Precedence (for every edge i→j):
s_j ≥ f_i + comm(i→j, device(i), device(j))

comm depends on device pair:
- 0 if same device
- overhead + bytes/bw if different device

4) Non-overlap (hard part):
For tasks i and k on same device p:
Either i finishes before k starts OR k finishes before i starts.

This is modeled using binary ordering variable y(i,k,p) and big-M constraints.

This makes MILP exact but expensive.
