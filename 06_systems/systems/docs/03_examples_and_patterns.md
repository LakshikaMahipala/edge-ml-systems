# Examples and patterns

1) Real-time vision (camera inference)
- typically latency-sensitive
- PCIe/SoC attachment is preferred
- network-attached only if batching or edge gateway design exists

2) Large transformer inference service
- compute heavy
- batching common
- network-attached accelerators are typical in production clusters

3) FPGA in lab setup via UART
- extremely IO-limited
- compute improvements won't show until IO improves
- good for functional proof, not throughput claims
