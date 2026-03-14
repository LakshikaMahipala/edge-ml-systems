from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from dag import DAG
from comm import LinkModel, comm_ms

@dataclass
class ScheduledTask:
    task_id: str
    device: str
    start: float
    end: float

def avg_cost(task_costs: Dict[str, float]) -> float:
    return sum(task_costs.values()) / len(task_costs)

def upward_ranks(dag: DAG, link: LinkModel) -> Dict[str, float]:
    memo: Dict[str, float] = {}

    def rank_u(tid: str) -> float:
        if tid in memo:
            return memo[tid]
        t = dag.tasks[tid]
        avg_w = avg_cost(t.costs)

        ch = dag.children(tid)
        if not ch:
            memo[tid] = avg_w
            return memo[tid]

        # average comm for edge is based on assuming different device (use link)
        best = 0.0
        for e in ch:
            avg_c = comm_ms(e.bytes, same_device=False, link=link)
            best = max(best, avg_c + rank_u(e.dst))
        memo[tid] = avg_w + best
        return memo[tid]

    for tid in dag.tasks:
        rank_u(tid)
    return memo

def find_earliest_slot(occupied: List[Tuple[float,float]], ready: float, dur: float) -> float:
    """
    occupied: list of (start,end) intervals for a device schedule, sorted by start.
    Return earliest start >= ready where [start,start+dur) fits.
    """
    if not occupied:
        return ready
    # check gap before first
    if ready + dur <= occupied[0][0]:
        return ready

    t = ready
    for (s,e) in occupied:
        if t + dur <= s:
            return t
        t = max(t, e)
    return t

def heft_schedule(dag: DAG, link: LinkModel) -> List[ScheduledTask]:
    ranks = upward_ranks(dag, link)
    order = sorted(dag.tasks.keys(), key=lambda t: ranks[t], reverse=True)

    # device schedules: device -> list of ScheduledTask
    sched: Dict[str, List[ScheduledTask]] = {d: [] for d in dag.devices}
    finish_time: Dict[str, float] = {}     # task -> finish
    assigned_dev: Dict[str, str] = {}      # task -> device

    for tid in order:
        best_dev = None
        best_start = None
        best_end = math.inf

        for dev in dag.devices:
            # ready time from parents
            ready = 0.0
            for pe in dag.parents(tid):
                p = pe.src
                p_end = finish_time[p]
                p_dev = assigned_dev[p]
                ready = max(ready, p_end + comm_ms(pe.bytes, same_device=(p_dev==dev), link=link))

            dur = dag.tasks[tid].costs[dev]
            occ = sorted([(x.start, x.end) for x in sched[dev]], key=lambda z: z[0])
            st = find_earliest_slot(occ, ready, dur)
            en = st + dur

            if en < best_end:
                best_end = en
                best_start = st
                best_dev = dev

        # commit assignment
        st = float(best_start)
        en = float(best_end)
        dev = str(best_dev)
        sched[dev].append(ScheduledTask(tid, dev, st, en))
        finish_time[tid] = en
        assigned_dev[tid] = dev

    # flatten and sort by start
    out = []
    for d in dag.devices:
        out.extend(sched[d])
    out.sort(key=lambda x: x.start)
    return out

def makespan(schedule: List[ScheduledTask]) -> float:
    return max((t.end for t in schedule), default=0.0)
