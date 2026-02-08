Repo Architecture Map (Pipeline view)

Stage A — Preprocess
- inference_bench : measures preprocess time
- video_demo/src/preprocess.py : real-time preprocess
- (Week 2) C++ preprocess modules (future)

Stage B — Inference
- inference_bench : PyTorch baseline
- tensorrt_bench  : TRT engines (FP32/FP16/INT8, DLA)
- transfer_bench  : copy overhead around GPU inference
- fpga/int8_mlp_v0 : FPGA compute kernel v0 (sim)
- fpga/protocol_uart : I/O protocol layer (sim + host scaffold)

Stage C — Postprocess + packaging
- inference_bench : postprocess timing
- video_demo/src/overlay.py : overlay + display loop

Systems analysis tools
- tools/latency_budget.py : copy+compute speedup bound
- pipeline_model/*        : pipeline bottleneck + queueing simulation
