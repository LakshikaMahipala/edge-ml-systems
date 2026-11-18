Transfer Budget Notes 

Key equation
Transfer time can be approximated as:
  T ≈ overhead + bytes / bandwidth

Why overhead matters
For small tensors (batch=1, small models), the fixed overhead dominates.
This is why simply “having a fast GPU” does not guarantee low latency.

What we added
- inference_bench/src/transfer_model.py
  - TransferLink(name, bandwidth_gbps, overhead_us)
  - estimate_transfer_ms(bytes, link)
- inference_bench/run_transfer_budget_demo.py

How to use it later
Example:
- python inference_bench/run_transfer_budget_demo.py --n 1 --c 3 --h 224 --w 224 --dtype_bytes 4

How it connects to Amdahl
Even if inference is accelerated, transfers and preprocessing can cap end-to-end gains.
This is the same principle as Amdahl’s Law applied to ML pipelines.
