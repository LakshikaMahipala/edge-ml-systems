Results Policy (JSON + Plots)

Why this policy exists
Benchmark results can bloat a repo and confuse readers unless managed.

Where results go
- Raw machine-readable outputs:
  inference_bench/results/*.json
- Human summary:
  docs/metrics.md
- Figures used in writeups:
  docs/plots/*.png

What to commit
- Only representative JSON files (e.g., one per milestone configuration)
- Only plots referenced in docs/week2_summary.md or milestone reports

What NOT to commit
- Hundreds of sweep outputs
- Large datasets
- Any private data or sensitive logs

Naming guidelines
- Prefer JSON filenames including model/device/batch/input when possible.
- Keep a short note in docs/metrics.md linking to the JSON filename.

Reproducibility requirement
Every committed plot must be regeneratable from committed JSON and scripts.
