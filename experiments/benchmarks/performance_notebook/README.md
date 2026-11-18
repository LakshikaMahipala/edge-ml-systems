Performance Notebook 

This module consolidates performance analysis:
- stage timing + IO
- Amdahl bounds
- queueing effects

Docs
- docs/performance_notebook.md
- docs/io_overhead_first_class.md
- docs/results_template.md

Scripts
- scripts/collect_results_stub.py
- scripts/make_summary_table.py

Run later
python scripts/collect_results_stub.py --backend pytorch_cpu --model resnet18 --device cpu --out results/pytorch_cpu_resnet18.json
python scripts/make_summary_table.py
