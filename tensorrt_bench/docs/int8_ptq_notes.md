INT8 PTQ (Post-Training Quantization) 

What INT8 PTQ means
- We keep the model weights as-is (no retraining).
- TensorRT calibrates activation ranges using a representative dataset.
- Quantization uses these ranges to map FP32 tensors -> INT8.

Why calibration exists
- INT8 needs scale factors per tensor (dynamic range).
- Calibration estimates these ranges from data.
- If you skip calibration, TensorRT may use random ranges and accuracy is not meaningful.

Two key artifacts
1) Calibration cache (e.g., calib.cache)
   - Stores learned ranges/scales
   - Reusable across builds (same model + preprocessing)

2) INT8 engine (.plan)
   - The optimized runtime artifact
   - Dependent on GPU architecture and TensorRT version

What to measure
- Latency: p50 / p99 (using trtexec or your runner)
- “Accuracy delta”:
  - True accuracy delta requires a dataset that matches the model labels (e.g., ImageNet val).
  - As a proxy (when you don’t have ImageNet), measure:
    - top-1 agreement between FP32 and INT8 engines on the same inputs
    - cosine similarity of logits

This repo implements the proxy comparison so you can run immediately later.
