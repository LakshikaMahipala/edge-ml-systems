Week 2 Integrated Pipeline (Concept)

            +-------------------+
image file ->| C++ preprocess    |-> NCHW float32 -> +------------------+
            | (OpenCV + SIMD)   |                   | PyTorch forward   |
            +-------------------+                   +------------------+
                                                        |
                                                        v
                                                   +-----------+
                                                   | postproc  |
                                                   +-----------+

Benchmark compares:
Path A: Python preprocess -> forward -> postproc
Path B: C++ preprocess -> forward -> postproc
