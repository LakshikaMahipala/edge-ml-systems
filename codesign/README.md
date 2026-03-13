HW/SW co-design blueprint (CoD-style workflow)

What this module provides:
- joint search space definition (architecture × hardware)
- objective and constraint definitions
- estimation ladder and failure modes
- stubs that enable tomorrow’s tiny co-design loop

Next:
- implement the co-design loop
- compute Pareto points
- report 2–3 final designs

Run later:
python src/run_codesign_loop.py --out_all results/codesign_points.jsonl
python src/select_points.py --in_all results/codesign_points.jsonl
