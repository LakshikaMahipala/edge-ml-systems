ONNX Runtime GPU on Jetson — Notes

Problem
- onnxruntime-gpu wheels are not always available on PyPI for aarch64.
- Users often struggle to get CUDA EP working on Jetson.  [oai_citation:5‡NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/getting-cuda-acceleration-for-onnx-on-jetson-orin-nano-with-jetpack-6-2/344745?utm_source=chatgpt.com)

Implication for our plan
- Our Week 5/6 comparisons will include:
  - ORT CPU baseline (easy and portable)
  - ORT GPU only when the correct Jetson-compatible wheel/build is available

Preferred approach
1) Use TensorRT for GPU acceleration on Jetson (first choice for performance)
2) Use ORT CPU as the portable baseline
3) Add ORT GPU only when we can install a Jetson-compatible build cleanly
