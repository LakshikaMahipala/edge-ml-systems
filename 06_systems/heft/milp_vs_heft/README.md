MILP scheduling vs HEFT (comparison)

What exists:
- MILP formulation docs (variables, constraints, makespan)
- Exact optimal baseline via brute-force enumeration (small DAGs)
- Comparison script: HEFT vs OPT gap (%)

Run later:
python src/compare_heft_optimal.py --dag schema/example_dag_small.json --bw_GBps 10 --overhead_ms 0.05
