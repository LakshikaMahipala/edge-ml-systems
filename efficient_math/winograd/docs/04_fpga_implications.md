FPGA implications (what maps well vs what hurts)

What maps well
- Offline filter transform: do it on CPU at compile time.
- Runtime becomes structured:
  input_transform -> elementwise_mult -> output_transform
- Elementwise mult is friendly: can be heavily pipelined and parallelized.

What hurts
- Transforms involve additions and constant multiplications (fractions).
  In fixed-point, you must choose scaling carefully.
- Intermediate storage increases: you may need more BRAM/buffers per tile.
- If your I/O is slow (UART), none of this matters for end-to-end latency.

FPGA “design knobs”
- tile parallelism (how many tiles processed simultaneously)
- channel parallelism (how many channels unrolled)
- II for transform pipelines
- BRAM buffering strategy to keep stream continuous
