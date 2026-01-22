Results Folder Policy

Purpose
This folder stores machine-generated benchmark outputs (JSON) that support reproducible tracking.

What goes here
- JSON outputs produced by scripts using --save_json
- Small text summaries copied from runs (optional)

Naming convention
Files are created automatically as:
{project}_{model}_{device}_{timestamp}.json

Commit policy
- Commit representative results when they support a report or milestone.
- Do NOT commit huge volumes of redundant results.
- Prefer: one baseline run + one optimized run + sweeps that are necessary for plots.

How results are used
- docs/metrics.md stores the summary table for humans.
- scripts/update_metrics_from_json.py can help extract a markdown row from JSON.
