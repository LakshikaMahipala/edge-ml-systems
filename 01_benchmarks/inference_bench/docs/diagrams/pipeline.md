# Integrated Pipeline (Concept)

```text
+-------------+      +-------------------------+      +------------------+      +-----------+
| image file  | ---> | C++ preprocess          | ---> | PyTorch forward  | ---> | postproc  |
| (disk)      |      | (OpenCV + SIMD)         |      | (model inference)|      | (top-k)   |
+-------------+      +-------------------------+      +------------------+      +-----------+
                          |
                          v
                   NCHW float32 tensor
Benchmark compares:
Path A: Python preprocess -> forward -> postproc
Path B: C++ preprocess -> forward -> postproc
