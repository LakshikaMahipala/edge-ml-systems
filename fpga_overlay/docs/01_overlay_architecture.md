# Overlay architecture (Conv/BN/ReLU engine)

High-level blocks:

1) DMA / Stream interface
- reads input activations from external memory (or UART stream)
- writes output activations back

2) On-chip buffers (BRAM)
- input tile buffer
- weight tile buffer
- partial sum (accumulator) buffer
- output tile buffer

3) Compute core
- systolic / MAC array or SIMD MAC lanes (P parallel MACs)
- supports INT8 accumulation to INT32

4) Post-ops unit
- BatchNorm (scale + bias)
- ReLU clamp

5) Controller / microcode
- reads layer descriptors
- schedules tile loops
- handles address generation

Key principle:
Dataflow is designed around reuse:
- keep weights or input tiles in BRAM as long as possible.
