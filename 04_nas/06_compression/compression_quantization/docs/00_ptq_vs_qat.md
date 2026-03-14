
## Quantization in one line
Replace floating-point tensors with low-precision integers (typically INT8) to reduce:
- memory footprint
- bandwidth
- compute cost (especially on edge accelerators / FPGA / DSP)

---

## PTQ: Post-Training Quantization
### What it is
Quantize a trained FP32 model **after training**, without retraining.

### How it works (typical)
1) Keep FP32 weights
2) Run calibration data to estimate activation ranges
3) Choose scales/zero-points
4) Convert weights/activations to int8 (or simulate)

### Pros
- fastest pipeline
- no retraining needed
- very practical for many CNNs

### Cons (the hidden problems)
- accuracy drop can be large for:
  - transformers
  - small models
  - layers with sensitive activation distributions
  - models with outliers

---

## QAT: Quantization-Aware Training
### What it is
Train the model while **simulating quantization** (“fake quant”) so it learns to be robust.

### Core trick
During forward pass:
- quantize/dequantize weights and activations with fake-quant ops
During backward pass:
- use STE (straight-through estimator) so gradients still flow

### Pros
- usually much smaller accuracy drop than PTQ
- essential for hard models / aggressive quantization

### Cons
- more training cost
- requires careful setup (fusing, observers, per-channel/per-tensor choices)

---

## What hardware actually cares about
1) **Operator coverage**
   - Can your backend run INT8 conv/matmul efficiently?
2) **Memory bandwidth**
   - INT8 helps a lot when bandwidth-bound
3) **Calibration stability**
   - wrong ranges → saturation → accuracy collapse
4) **Per-channel weight quantization**
   - especially important for conv/linear weights

---

## Your internship-level rule
- Start with PTQ if you want speed and the model is “easy”.
- Use QAT when:
  - PTQ accuracy drop is unacceptable
  - model has heavy tails/outliers
  - you target strict INT8 accuracy requirements
