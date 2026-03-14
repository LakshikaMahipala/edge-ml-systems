# Tiny co-design loop (FPGA-aware, proxy-based)

## Goal
Jointly search:
- architecture knobs (block, width, depth)
- hardware knobs (parallelism P, tile, II_factor)

Objectives:
- maximize acc_proxy
- minimize latency_proxy
- minimize energy_proxy

## Method
1) Enumerate joint space A×H
2) Compute proxy metrics
3) Pareto filter
4) Select 3 representative Pareto points:
   - fast
   - balanced
   - accurate

## Evidence policy
All values here are proxies until real FPGA measurements are plugged in.
This is still valuable because co-design decisions require tradeoff structure.

## Outputs (run later)
- results/codesign_points.jsonl
- results/pareto_points.jsonl
- results/selected_points.json
- report/tables/template_pareto_table.csv (to be filled)
