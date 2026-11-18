Jetson Stack Overview (Week 6 Day 1)

Mental model (layers)
1) Jetson Linux (L4T) + Ubuntu base OS
2) CUDA-X libraries: CUDA, cuDNN, TensorRT, VPI
3) Inference runtimes:
   - TensorRT (highest performance on NVIDIA GPU)
   - ONNX Runtime (portable runtime; acceleration depends on Execution Provider)
   - PyTorch (dev-friendly; usually slower unless compiled/optimized)
4) Application layer:
   - preprocessing (decode/resize/normalize)
   - inference
   - postprocessing
5) Deployment packaging:
   - native install (apt)
   - containers (NGC l4t-* images)

Official architecture reference
- NVIDIA’s Jetson software architecture describes CUDA/CUDA-X + frameworks above Jetson Linux.  [oai_citation:0‡NVIDIA Docs](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/AR/JetsonSoftwareArchitecture.html?utm_source=chatgpt.com)
