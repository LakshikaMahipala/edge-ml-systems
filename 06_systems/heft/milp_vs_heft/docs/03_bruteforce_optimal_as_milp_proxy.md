# Why brute-force optimal works as a MILP proxy (for small DAGs)

We don't always have a MILP solver installed (Gurobi/CPLEX).

For small DAGs (<= 8 tasks), we can:
- enumerate device assignments
- for each assignment, compute best schedule using an exact simulation
- pick the minimum makespan

This gives the true optimum for small cases.
We use it to compare HEFT vs optimal.

This is valid for Week 16 learning and portfolio evidence.
