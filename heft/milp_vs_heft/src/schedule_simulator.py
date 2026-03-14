from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
from dag_loader import DAG, Edge

@dataclass
class SimTask:
    task_id: str
    device: str
    start: float
    end: float

def comm_cost_ms(bytes_: int, same: bool, bw_GBps: float, overhead_ms: float) -> float:
    if same:
        return 0.0
    bw_Bps = bw_GBps * (1024**3)
    return overhead_ms + (bytes_ / bw_Bps) * 1e3

def simulate_schedule(
    dag: DAG,
    assign: Dict[str, str],
    bw_GBps: float = 10.0,
    overhead_ms: float = 0.05
) -> Tuple[List[SimTask], float]:
    """
    Exact earliest-start simulation for a FIXED assignment.
    Uses list scheduling in topological order with device availability.
    """
    topo = dag.topo_sort()
    dev_free: Dict[str, float] = {d: 0.0 for d in dag.devices}
    finish: Dict[str, float] = {}
    sched: List[SimTask] = []

    for t in topo:
        dev = assign[t]
        # dependency ready time
        ready = 0.0
        for e in dag.parents(t):
            p = e.src
            p_dev = assign[p]
            ready = max(ready, finish[p] + comm_cost_ms(e.bytes, same=(p_dev==dev), bw_GBps=bw_GBps, overhead_ms=overhead_ms))
        start = max(ready, dev_free[dev])
        dur = dag.tasks[t].costs[dev]
        end = start + dur
        dev_free[dev] = end
        finish[t] = end
        sched.append(SimTask(t, dev, start, end))

    makespan = max((x.end for x in sched), default=0.0)
    return sched, makespan
