# Comparison protocol: HEFT vs optimal

For a given DAG:
1) Run HEFT to get schedule and makespan
2) Run optimal brute-force scheduler to get optimal makespan
3) Compute gap:
gap% = (HEFT - OPT) / OPT * 100

Repeat on a few toy DAGs.

Interpretation:
- small gap: HEFT is close to optimal
- large gap: HEFT struggles; investigate comm cost or critical path structure
