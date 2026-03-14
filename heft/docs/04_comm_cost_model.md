# Communication cost model

If two dependent tasks are placed on different devices,
we pay a transfer cost for the tensor on that edge.

Proxy model:
comm_time = overhead + tensor_bytes / bandwidth

Special case:
If tasks are on same device, comm_time = 0.
