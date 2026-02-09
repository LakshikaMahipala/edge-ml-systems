Latency vs throughput vs energy (discussion)

Latency
- time per item (p50/p99).
In UART FPGA systems, latency is often dominated by serialization time.

Throughput
- items/s.
Even if FPGA compute is fast, UART caps throughput to:
throughput <= (baud_bytes_per_sec / bytes_per_item)

Energy (first-principles discussion)
We cannot measure energy today, but we can reason:
- MCU/NPU/FPGA energy depends heavily on data movement.
- Sending bytes over UART/PCIe costs energy too.
- “Compute savings” are irrelevant if you spend more energy moving data.

Expected outcome for our Week 8 block
- FPGA compute cycles are small.
- UART I/O dominates time and likely dominates energy per inference.
- Therefore, the correct engineering move is to improve interface or increase workload per transfer.

Researcher takeaway
This is exactly why platform thinking matters:
- kernels alone do not define performance
- system I/O + scheduling define the real envelope
