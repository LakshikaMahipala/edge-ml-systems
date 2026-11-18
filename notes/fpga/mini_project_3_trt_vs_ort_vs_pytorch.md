Mini-project 3 — PyTorch vs ONNX Runtime vs TensorRT (+ FPGA appendix)

Goal
Compare inference performance across:
1) PyTorch eager inference
2) ONNX Runtime (CPU or CUDA EP later)
3) TensorRT (FP32/FP16/INT8)

We measure (
- End-to-end latency: p50/p99
- Breakdown: preprocess / inference / postprocess
- Throughput: images/sec at batch=1 (and optionally batch>1)
- Accuracy proxy:
  - top-1 / top-5 (if labels available)
  - FP32 vs INT8 output drift proxy (agreement + cosine)

Models (baseline)
- resnet18
- mobilenet_v3_small
- efficientnet_b0

How to reproduce 

A) PyTorch (inference_bench)
- Run run_pytorch_benchmark.py and save JSON outputs.
- Record p50/p99 from PerfSummary.

B) ONNX Runtime (Week 3 mini project 1B)
- Export ONNX, run ORT session, record p50/p99.
- Use same input tensor shape and preprocessing.

C) TensorRT (tensorrt_bench)
- Export ONNX -> build engine (fp32, fp16) -> benchmark via trtexec logs
- Parse logs to JSON (p50/p99)
- Build INT8 engine with calibrator -> compare drift vs fp32

Results table (fill after runs)

Model: resnet18 (1x3x224x224)

Backend | Precision | Device | p50 (ms) | p99 (ms) | Throughput (qps) | Notes
- PyTorch | FP32 | CPU/GPU | | | |
- ORT     | FP32 | CPU/GPU | | | |
- TRT     | FP32 | GPU     | | | |
- TRT     | FP16 | GPU     | | | |
- TRT     | INT8 | GPU     | | | | calib cache used

Repeat for other models.

Key analysis questions (answer after data)
1) Which backend wins on p50? On p99? Why?
2) How much do fusions help (TRT FP16 vs FP32)?
3) Does batch=1 differ from batch>1 (throughput vs latency trade)?
4) Where does time go? (preprocess vs GPU compute vs postprocess)
5) For INT8: what is the drift proxy? Is it acceptable?

FPGA appendix — where FPGA would sit (budget model)
We treat FPGA as an accelerator for part of the pipeline.
Total latency:
T_total = T_pre + T_copy_to_accel + T_compute_accel + T_copy_back + T_post

On early FPGA demos (UART-based):
- T_copy_to_accel and T_copy_back are large (UART bandwidth is low).
- Therefore speedups are capped by I/O (Amdahl-style).

What we will do 
- Use UART echo + fixed-point lib as foundations.
- Implement INT8 MLP/linear layer in FPGA (Week 6).
- Measure host->FPGA->host latency, then compare to CPU/GPU.
- Use tools/latency_budget.py to compute theoretical speedup bounds.

Artifacts in repo
- inference_bench/ : PyTorch + component breakdown
- tensorrt_bench/  : TRT build + parse + INT8 calibration scripts
- fpga/            : UART echo + fixed-point primitives
- tools/latency_budget.py : budget model calculator
