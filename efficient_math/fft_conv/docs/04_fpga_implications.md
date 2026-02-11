FPGA implications

FFT is not “just math”; it is a hardware architecture problem.

Pros
- structured butterfly stages -> pipeline-friendly
- high throughput if streaming is designed well

Cons
- requires significant on-chip buffering (BRAM) for stages
- complex control and addressing
- fixed-point scaling at each stage is non-trivial
- for small kernels (3x3), it is the wrong tool

Practical takeaway
FFT conv is FPGA-relevant mainly for:
- long 1D signals (audio/radar/vibration)
- large-kernel filtering
- when you can amortize transforms and reuse FFT(h)
