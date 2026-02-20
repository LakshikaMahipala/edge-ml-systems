FPGA latency regularizer validation (scaffold)

Purpose:
Validate that our latency proxy preserves ranking vs real FPGA kernel measurements.

What to do later:
1) Implement/measure 5–10 FPGA kernel cases (see src/kernels_to_measure.py).
2) Save measurements in data/measurements.jsonl (same fields as template).
3) Run validation:
   python src/validate_latency.py --meas data/measurements.jsonl

Outputs:
- data/validation.json with Spearman correlation + MAPE.
