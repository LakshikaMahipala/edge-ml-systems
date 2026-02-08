Jetson + FPGA + Video Demo

Jetson track (docs only, runnable later)
- Jetson stack overview
- JetPack version matrix
- ORT GPU constraints notes
- Deployment playbook (native vs container)

FPGA track (sim-ready)
- INT8 MLP v0 kernel (RTL + TB + sim script)
- UART protocol layer (CRC framing + loopback TB)
- Host-side UART protocol scaffold (python)

Systems track (copy overhead)
- transfer_bench: pageable vs pinned H2D/D2H microbenchmark (run later on GPU)
- copy-time estimator script
- zero-copy vs pinned memory theory notes

Mini-project 4 (host app)
- video_demo: webcam/video classifier scaffold
- overlay: latency + FPS + rolling p50/p99
- FPGA offload point documented in diagram

Next week (Week 7)
- Pipeline + queueing model
- FPGA: pipeline the MLP datapath (throughput vs latency)
- Host async driver + backpressure behavior
- Amdahl deep dive with measured serial fractions
