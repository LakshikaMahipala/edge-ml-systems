HEFT scheduling for CPU/GPU/FPGA DAGs

What exists:
- DAG schema (tasks, edges, device costs)
- HEFT implementation:
  - upward rank priority
  - insertion policy
  - device assignment minimizing earliest finish time
- runner that exports schedule + makespan

Run later:
python src/run_heft.py --dag schema/example_dag.json --bw_GBps 10 --overhead_ms 0.05
