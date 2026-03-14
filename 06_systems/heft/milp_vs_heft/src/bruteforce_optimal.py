from __future__ import annotations
import itertools
from typing import Dict, Any, Tuple, List
from dag_loader import DAG
from schedule_simulator import simulate_schedule

def optimal_schedule_bruteforce(
    dag: DAG,
    bw_GBps: float = 10.0,
    overhead_ms: float = 0.05,
    max_tasks: int = 10
) -> Tuple[Dict[str,str], float]:
    task_ids = list(dag.tasks.keys())
    if len(task_ids) > max_tasks:
        raise ValueError(f"Too many tasks for brute-force ({len(task_ids)}). Reduce DAG size.")

    best_ms = float("inf")
    best_assign = None

    # assignments: product of devices for each task
    for devs in itertools.product(dag.devices, repeat=len(task_ids)):
        assign = {task_ids[i]: devs[i] for i in range(len(task_ids))}
        _, ms = simulate_schedule(dag, assign, bw_GBps=bw_GBps, overhead_ms=overhead_ms)
        if ms < best_ms:
            best_ms = ms
            best_assign = assign

    return best_assign, best_ms
