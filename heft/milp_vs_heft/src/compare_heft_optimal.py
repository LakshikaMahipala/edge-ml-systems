from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math

from dag_loader import load_dag_json, DAG
from schedule_simulator import comm_cost_ms
from bruteforce_optimal import optimal_schedule_bruteforce, simulate_schedule

def avg_cost(costs: Dict[str, float]) -> float:
    return sum(costs.values()) / len(costs)

def upward_ranks(dag: DAG, bw_GBps: float, overhead_ms: float) -> Dict[str, float]:
    memo: Dict[str, float] = {}

    def rank_u(tid: str) -> float:
        if tid in memo:
            return memo[tid]
        avg_w = avg_cost(dag.tasks[tid].costs)
        ch = dag.children(tid)
        if not ch:
            memo[tid] = avg_w
            return memo[tid]
        best = 0.0
        for e in ch:
            avg_c = comm_cost_ms(e.bytes, same=False, bw_GBps=bw_GBps, overhead_ms=overhead_ms)
            best = max(best, avg_c + rank_u(e.dst))
        memo[tid] = avg_w + best
        return memo[tid]

    for t in dag.tasks:
        rank_u(t)
    return memo

def heft_assignment(dag: DAG, bw_GBps: float, overhead_ms: float) -> Tuple[Dict[str,str], float]:
    ranks = upward_ranks(dag, bw_GBps, overhead_ms)
    order = sorted(dag.tasks.keys(), key=lambda t: ranks[t], reverse=True)

    assign: Dict[str, str] = {}
    finish: Dict[str, float] = {}
    dev_free: Dict[str, float] = {d: 0.0 for d in dag.devices}

    # simple insertion-less EFT (enough for comparison baseline)
    for t in order:
        best_dev = None
        best_end = float("inf")
        best_start = 0.0
        for dev in dag.devices:
            ready = 0.0
            for e in dag.parents(t):
                p = e.src
                p_dev = assign[p]
                ready = max(ready, finish[p] + comm_cost_ms(e.bytes, same=(p_dev==dev), bw_GBps=bw_GBps, overhead_ms=overhead_ms))
            start = max(ready, dev_free[dev])
            end = start + dag.tasks[t].costs[dev]
            if end < best_end:
                best_end = end
                best_start = start
                best_dev = dev
        assign[t] = best_dev
        dev_free[best_dev] = best_end
        finish[t] = best_end

    # compute makespan by simulating in topo order for consistency
    _, ms = simulate_schedule(dag, assign, bw_GBps=bw_GBps, overhead_ms=overhead_ms)
    return assign, ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dag", type=str, default="schema/example_dag_small.json")
    ap.add_argument("--bw_GBps", type=float, default=10.0)
    ap.add_argument("--overhead_ms", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="results/compare.json")
    args = ap.parse_args()

    dag = load_dag_json(args.dag)

    a_heft, ms_heft = heft_assignment(dag, args.bw_GBps, args.overhead_ms)
    a_opt, ms_opt = optimal_schedule_bruteforce(dag, args.bw_GBps, args.overhead_ms, max_tasks=10)

    gap = 0.0 if ms_opt == 0 else (ms_heft - ms_opt) / ms_opt * 100.0

    out = {
        "dag": args.dag,
        "bw_GBps": args.bw_GBps,
        "overhead_ms": args.overhead_ms,
        "makespan_heft_ms": ms_heft,
        "makespan_opt_ms": ms_opt,
        "gap_percent": gap,
        "assign_heft": a_heft,
        "assign_opt": a_opt,
        "note": "OPT computed by brute-force enumeration for small DAGs (exact)."
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote:", args.out)
    print("HEFT:", ms_heft, "OPT:", ms_opt, "gap%:", gap)

if __name__ == "__main__":
    main()
