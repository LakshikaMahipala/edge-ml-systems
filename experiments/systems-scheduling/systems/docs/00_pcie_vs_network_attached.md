# PCIe vs network-attached accelerators 

## PCIe-attached accelerator
Accelerator sits on the same machine as CPU and is connected via PCIe.
Examples:
- desktop GPU
- many server GPUs
- PCIe-attached FPGA cards

Data path:
CPU memory → DMA over PCIe → accelerator memory → compute → return over PCIe

Pros:
- low latency vs network
- high bandwidth (Gen4/Gen5)
- simpler programming model

Cons:
- host CPU is still involved for orchestration
- PCIe transfers can dominate if per-inference payload is large or batch is small

## Network-attached accelerator
Accelerator is a separate node on the network fabric.
Examples:
- GPU/FPGA in a different server accessed via RDMA
- “accelerator appliance” behind a service API

Data path:
CPU → network stack / RDMA → remote accelerator → return over network

Pros:
- resource pooling and multi-tenant sharing
- can scale out and schedule across many accelerators
- useful when compute is huge and amortizes network cost

Cons:
- higher latency per request (network RTT)
- jitter (p99/p999)
- serialization + RPC overhead
