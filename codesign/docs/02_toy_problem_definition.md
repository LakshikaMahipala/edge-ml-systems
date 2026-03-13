# Toy co-design problem definition 

We will implement a tiny co-design loop on FPGA proxy:

Architecture choices (A):
- block type: {conv3, conv5} or {mbconv3, mbconv5}
- width: {16, 24}
- depth: {2, 3}

Hardware choices (H):
- unroll factor P: {8, 16, 32, 64}
- tile size T: {8, 16, 32} (conceptual)
- II penalty factor: {1.0, 1.2} (proxy for pipelining quality)

Goal:
Find Pareto points:
- maximize acc_proxy(A)
- minimize latency_proxy(A,H)
- minimize energy_proxy(A,H) (optional)

We will report 2–3 Pareto points.
