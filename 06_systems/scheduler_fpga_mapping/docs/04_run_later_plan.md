# Run-later plan

1) Fill latency registry with real measured FPGA kernel points later.
2) Generate a pipeline DAG JSON from the pipeline spec + registry.
3) Run:
   - HEFT schedule
   - OPT (bruteforce) schedule for small DAGs
4) Compare makespan and placements.

Commands (later):
python src/run_schedulers_stub.py
