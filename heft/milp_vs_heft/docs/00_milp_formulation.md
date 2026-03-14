# MILP scheduling formulation 

We want an optimal schedule for a heterogeneous DAG.

Inputs:
- tasks i ∈ V
- devices p ∈ P
- compute time w(i,p)
- communication time c(i→j) if devices differ

Decision variables:
- x(i,p) ∈ {0,1}: task i assigned to device p
- s_i ≥ 0: start time of task i
- C_max ≥ 0: makespan

MILP objective:
minimize C_max

Constraints encode:
- each task assigned to exactly one device
- precedence constraints with comm cost
- non-overlap constraints for tasks on same device
