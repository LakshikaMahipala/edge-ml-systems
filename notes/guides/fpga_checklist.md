# FPGA checklist (implementation readiness)

When implementing any math trick on FPGA, confirm:

## 1) IO regime
- UART? then end-to-end will be IO dominated
- PCIe/Ethernet? then compute optimizations matter

## 2) Precision plan
- FP32 / FP16 / INT8?
- For INT8: define scaling strategy and saturation points

## 3) Buffering plan (BRAM)
- what tile buffers exist?
- what intermediate tensors must be stored?
- can buffers stream without stalls?

## 4) Pipeline plan
- define initiation interval (II)
- identify stages that dominate (transform vs MAC)

## 5) Measurement plan
- cycle counters in RTL
- end-to-end p50/p99 on host
- throughput under continuous stream

## 6) Validation
- golden model (Python) comparison
- error metrics and tolerance
