DLA basics + selection logic 

What DLA is (in one sentence)
DLA (Deep Learning Accelerator) is a dedicated inference processor on many NVIDIA SoCs; TensorRT can execute supported parts of a network on DLA and the rest on GPU. (It supports only a subset of TensorRT layers.)  [source: TensorRT capabilities + Working with DLA docs]

Why you use DLA
- Power efficiency / freeing GPU for other tasks
- Deterministic offload of CNN-ish operators when compatible

Hard constraints (the “gotchas”)
- DLA supports only a subset of operators and parameter ranges.
- If your model includes unsupported ops, you must either:
  (A) fall back to GPU for those layers, or
  (B) redesign the model to be DLA-compatible.

Precision reality
- DLA is typically used with FP16 and INT8 paths (platform dependent).
- INT8 requires proper calibration (we built that in Week 5).

Selection logic (the policy we use)
We choose DLA only if all of these are true:
1) Device has DLA cores available.
2) Model has high DLA operator coverage (mostly conv / FC / activations / pooling / BN).
3) The expected benefit is meaningful under our latency budget:
   - If copy/IO dominates, DLA offload won’t matter.
4) We have a safe fallback plan:
   - If any layer can’t run on DLA, the engine can run those layers on GPU (allow GPU fallback),
     OR we deliberately build DLA-only and accept failure as a “compatibility test”.

Decision table
Case 1: Want max reliability (“ship it”)
- Enable DLA where possible + allow GPU fallback.
- This prevents build/runtime failures due to one unsupported layer.

Case 2: Want strict DLA validation (“research/compatibility mode”)
- Build DLA-only (no fallback).
- If it fails, you learn exactly what prevents full offload.

Case 3: Want best latency
- Benchmark GPU-only vs DLA(+fallback).
- Pick by p50/p99 + system power target.
- Do not assume DLA is always faster; sometimes GPU can be faster, especially for small batches.

What we will do later (Week 6+)
- On a Jetson device: attempt DLA build + fallback.
- Record the final layer placement and p50/p99.
- Add results to Mini-project tables.
