Plots Folder

Purpose
This folder stores generated figures that support milestone reports.

What goes here
- Latency plots (p50/p99)
- Sweep plots (latency vs batch/input size)
- Any figure referenced in docs/mini_project_0_report.md or weekly summaries

Commit policy
- Only commit plots that support a milestone (v0.1, v0.2, etc.)
- Do not commit large numbers of redundant plots

How to generate
- python scripts/plot_results.py --results_dir inference_bench/results --out_dir docs/plots
