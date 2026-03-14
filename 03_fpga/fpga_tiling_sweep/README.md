FPGA Tiling Sweep Dataset 

What it is
A sweep generator for FPGA kernel design knobs (UNROLL, II) and shapes,
labeled with our FPGA latency estimator v0.

Why
This mimics the schedule->latency datasets used in ML compilers.
Later we’ll replace estimated labels with measured labels.

Run later
python src/generate_fpga_sweep_points.py --cfg examples/sweep_config.yaml --out_jsonl data/sweep_points.jsonl
python src/build_fpga_sweep_dataset.py --in_jsonl data/sweep_points.jsonl --out_csv data/fpga_sweep_dataset.csv
