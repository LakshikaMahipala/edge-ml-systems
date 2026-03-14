Map FPGA kernels into scheduler model (HEFT/OPT)

What exists:
- latency registry (per kernel per device latency values; proxy or measured)
- pipeline spec → DAG generator (inject device costs)
- run plan to schedule with HEFT and compare with OPT baseline

Run later:
python src/pipeline_to_dag.py
python ../heft_week16/src/run_heft.py --dag results/pipeline_dag.json --out results/heft_schedule.json
python ../milp_vs_heft_week16/src/compare_heft_optimal.py --dag results/pipeline_dag.json --out results/compare.json
