Video inference pipeline + future FPGA offload point

[Camera/Video Source]
        |
        v
[CPU Preprocess: decode/resize/normalize]  <-- still CPU in our plan
        |
        v
[Inference Backend]
  - PyTorch (now)
  - ORT (later)
  - TRT (later)
  - FPGA kernel (future offload point)
        |
        v
[CPU Postprocess + Overlay]
  - top-k decode
  - FPS / latency overlay
        |
        v
[Display / Save]

Where FPGA fits
- FPGA replaces a portion of inference (e.g., FC/MLP or conv block)
- Host sends tensors -> FPGA -> receives logits
- Early stage UART: I/O dominates (use latency_budget.py)
- Later stage PCIe / faster link: compute offload becomes meaningful
