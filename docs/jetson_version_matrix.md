Jetson Version Matrix (practical)

Two major tracks exist today:

Track A — JetPack 6.x (Orin-family mainstream)
- JetPack 6.2.1 release notes list:
  CUDA 12.6, TensorRT 10.3, cuDNN 9.3, VPI 3.2, DLA 3.1.  [oai_citation:1‡NVIDIA Docs](https://docs.nvidia.com/jetson/jetpack/release-notes/?utm_source=chatgpt.com)
- This is the “most common real-world Jetson Orin deployment baseline” right now.

Track B — JetPack 7.x (newer platform line / Ubuntu 24.04 / Kernel 6.8)
- JetPack page indicates JetPack 7 is built on Ubuntu 24.04 LTS and Linux kernel 6.8 and is part of a newer modular stack.  [oai_citation:2‡NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack?utm_source=chatgpt.com)
- JetPack 7.0 archive notes show Ubuntu 24.04 + Kernel 6.8 and support for Jetson AGX Thor/T5000 module.  [oai_citation:3‡NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.0?utm_source=chatgpt.com)

Important constraint
- JetPack availability depends on the exact Jetson device family (e.g., Orin Nano vs Thor), and forum discussions show people asking about JetPack 7 support on Orin Nano specifically.  [oai_citation:4‡NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/jetpack-7-x-support-for-jetson-orin-nano-tensorrt-10-5/359738?utm_source=chatgpt.com)

Rule you follow in this repo
- We write scripts that are “JetPack-agnostic”.
- We document a known-good baseline (JetPack 6.2.1 family) and keep JetPack 7 as a forward-looking track.
