# When network-attached accelerators win

Network-attached makes sense when:
- compute per request is large (long kernel time)
- batching is possible (amortize transfer/RPC overhead)
- you need pooling (share accelerators across many users)
- your system is throughput-oriented, not ultra-low-latency

PCIe-attached is preferred when:
- strict latency targets (interactive)
- small payload but frequent requests
- tight p99 tail latency requirements
