# Mini-project 13 — HW-aware NAS + Co-design

## 1. Overview
This mini-project demonstrates a complete workflow for hardware-aware architecture selection:
- multi-objective optimization (Pareto frontier)
- latency prediction (graph-based spec + baselines)
- co-design (joint search over model + hardware knobs)
- selection of Pareto-optimal design points

## 2. Multi-objective optimization (Pareto)
We define objectives:
- accuracy (maximize)
- latency (minimize)
- energy (minimize)

We filter dominated solutions and report a Pareto frontier.

Repo artifact:
- pareto_week15/

## 3. Latency prediction (scaling NAS evaluation)
We define a graph representation and feature spec for a GNN latency predictor.
Before GNN, we ship baselines:
- regression MLP on aggregated graph features
- pairwise ranking model for few-shot ordering

Repo artifacts:
- gnn_latency_week15/
- latency_predictor_week15/

## 4. Co-design workflow (CoD-style)
We define joint search space:
candidate = (arch, hw)

arch knobs:
- block, width, depth

hw knobs:
- unroll factor P, tile size, II_factor

Repo artifact:
- codesign_week15/

## 5. Tiny co-design loop (proxy-based)
We enumerate candidates, compute proxies, and extract Pareto points:
- fast point
- balanced point
- accurate point

## 6. Results tables (to be populated after local runs)
- report/tables/template_summary_table.csv
- report/tables/template_pareto_points_table.csv
- report/tables/template_latency_predictor_table.csv

## 7. Evidence policy
See docs/01_evidence_policy.md

## 8. Run-later instructions
See docs/02_how_to_reproduce_run_later.md
