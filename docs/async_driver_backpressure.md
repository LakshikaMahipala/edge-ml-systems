Async UART driver + back-pressure 

Why async?
Synchronous UART code blocks:
- waiting for bytes
- decoding frames
- timing out
This creates jitter and destroys p99 latency.

Design goals
1) Reader thread always drains UART -> prevents OS/UART buffer overflow.
2) Decoder runs continuously -> extracts complete frames.
3) Main thread sends requests and awaits responses WITHOUT blocking reads.
4) Bounded queue -> back-pressure:
   - if the system is overwhelmed, we either block sender or drop requests (configurable).
5) Stability metrics:
   - dropped frames
   - CRC failures
   - timeouts
   - response latency distribution (p50/p99)

Back-pressure policy options
A) BLOCK
- If queue full, sender blocks until space.
- Best for correctness; may reduce throughput.

B) DROP_OLDEST
- Keep newest data (good for real-time streaming where old frames are useless).

C) DROP_NEWEST
- Keep ordered correctness; reject new work when overloaded.

In later weeks
- This async design maps to PCIe/NIC style drivers too.
- Same structure applies to GPU pipelines (producer/consumer).
