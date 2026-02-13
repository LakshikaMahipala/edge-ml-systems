# When FPGA + BNN makes sense (and when it doesn't)

## FPGA + BNN makes sense when:

### A) You are bandwidth-limited
- weights and activations are large relative to compute
- moving FP32 data dominates energy and time
BNN reduces memory by ~32× and IO pressure heavily.

### B) Your compute is logic-friendly
- bitwise ops + popcount map well to LUTs and carry chains
- DSPs are not required for multipliers

### C) You can pipeline for throughput
- streaming inference where steady-state throughput matters
- you can accept some latency but need high FPS

### D) Your accuracy target is compatible
- classification tasks with margin
- robust sensor tasks
- tasks where FP32 accuracy is not mission-critical

---

## FPGA + BNN is a trap when:

### 1) IO dominates (UART regime)
If your host link is slow:
- end-to-end latency dominated by transfer, not compute
BNN compute improvements won't show.

### 2) Your task needs high precision
- regression, fine-grained detection, transformer language modeling
BNN accuracy drop may be unacceptable.

### 3) You cannot afford retraining/architecture tuning
BNN typically needs:
- BN, wider layers, scaling tricks
- careful training schedule

### 4) GPU INT8 already solves it
If you have TensorRT on Jetson/GPU:
- INT8 Tensor Cores may beat FPGA effort for time-to-result

---

## Practical comparison rule (intern-ready)
If you want speed quickly: GPU INT8 (TensorRT).
If you want hardware credibility + low-power specialization: FPGA + BNN.
