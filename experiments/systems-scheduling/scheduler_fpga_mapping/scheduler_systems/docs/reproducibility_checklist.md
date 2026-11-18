# reproducibility checklist

## Inputs to record
- device definitions (cpu/gpu/fpga)
- link model (bandwidth, overhead)
- latency registry (kernel_id × device latency)
- pipeline spec (tasks + tensor bytes)

## Scripts to run (later)
1) Transfer budget:
systems_week16/src/budget_calculator.py

2) Build pipeline DAG:
scheduler_fpga_mapping_week16/src/pipeline_to_dag.py

3) HEFT schedule:
heft_week16/src/run_heft.py

4) OPT baseline:
milp_vs_heft_week16/src/compare_heft_optimal.py

5) Populate report:
miniproject14_scheduler_systems/report/tables/*.csv

## Evidence rules
- label any non-measured numbers as proxy
- include p50/p99 targets where relevant
- never claim speedup without full end-to-end measurement context
