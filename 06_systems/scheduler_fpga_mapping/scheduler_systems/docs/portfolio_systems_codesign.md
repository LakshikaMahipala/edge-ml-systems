# Week 16 Portfolio Section — Systems Co-design + Scheduling

## What I built
A complete systems-level workflow for reasoning about end-to-end accelerator performance:

1) Attachment & transfer modeling
- PCIe vs network-attached accelerator analysis
- copy-cost budgeting to quantify transfer-dominated regimes

2) FPGA overlay concept
- config-driven Conv/BN/ReLU overlay design
- layer descriptor ISA (JSON contract), memory map, tiling/dataflow plan
- proxy cycle model for compute-vs-memory boundedness

3) Heterogeneous scheduling
- HEFT scheduling implementation for CPU/GPU/FPGA DAGs
- exact baseline via brute-force optimal scheduling for small DAGs (MILP proxy)
- comparison harness that reports makespan and HEFT-vs-OPT gap

4) Mapping FPGA kernels into schedules
- latency registry schema (kernel_id × device → latency)
- pipeline→DAG generator that injects per-device costs
- run-later plan to populate the registry with measured FPGA kernel points

## Why it matters
Kernel optimization alone does not guarantee system speedup.
End-to-end performance is constrained by:
- data movement
- orchestration and dependency placement
- tail latency
- scheduling decisions

This work provides the tooling and artifacts needed to make hardware claims with evidence.

## Evidence policy
All latencies/bandwidths are labeled:
- "proxy" until measured
Schedules are only as valid as the latency registry that generated them.

## Where to look in the repo
- systems_week16/ (PCIe vs network-attached, budget calculator)
- fpga_overlay_week16/ (overlay ISA + tiling + cycle model)
- heft_week16/ (HEFT scheduler)
- milp_vs_heft_week16/ (MILP formulation + brute-force OPT baseline)
- scheduler_fpga_mapping_week16/ (latency registry + pipeline→DAG)
- miniproject14_scheduler_systems/ (consolidated report)
