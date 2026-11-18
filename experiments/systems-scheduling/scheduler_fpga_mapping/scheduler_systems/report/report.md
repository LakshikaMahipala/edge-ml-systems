# Mini-project 14 — Heterogeneous scheduling + systems co-design (Week 16)

## 1. Overview
This mini-project demonstrates that accelerator performance is a systems problem:
- attachment model (PCIe vs network-attached)
- reusable FPGA overlay execution model
- heterogeneous scheduling (HEFT) vs optimal baseline (OPT/MILP)
- mapping FPGA kernel latencies into scheduling inputs

## 2. Attachment models (Week 16 Day 1)
We build a transfer budget model:
T_total = pre + H2D + compute + D2H + post

We compute transfer fraction and identify transfer-dominated regimes.

## 3. FPGA overlay concept 
We define a Conv/BN/ReLU overlay with:
- layer descriptor ISA (JSON contract)
- memory map
- tiling/dataflow strategy
- cycle model proxy

## 4. Scheduling algorithms
### 4.1 HEFT (Week 16 Day 3)
- upward rank priority
- earliest finish time assignment
- insertion policy (in HEFT module)

### 4.2 OPT/MILP baseline (Week 16 Day 4)
- MILP formulation documented
- brute-force optimal scheduler implemented for small DAGs
- HEFT vs OPT gap computed

## 5. FPGA kernel mapping (Week 16 Day 5)
We define a latency registry:
kernel_id × device → latency_p50_ms

We generate pipeline DAGs by injecting per-device costs, then schedule.

## 6. Results tables (to be populated after runs)
- systems budget table
- schedule makespan comparison table
- latency registry points table

## 7. Evidence policy
See docs/01_evidence_policy.md

## 8. Run-later instructions
See docs/02_run_later_plan.md
