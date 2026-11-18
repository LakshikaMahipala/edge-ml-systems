Command stream + scheduler model (v0)

Command model
A model execution is a list of commands.
Each command is an operator invocation with:
- op type
- input buffer address + shape
- output buffer address + shape
- weights address (if needed)
- quant params (SHIFT for v0)

Minimal command structure
{
  "op": "INT8_FC",
  "in_addr": 0x0000,
  "out_addr": 0x1000,
  "w_addr": 0x2000,
  "params": {"IN": 8, "OUT": 4, "SHIFT": 7}
}

Scheduler model
We use a host-driven scheduler:
- host pushes commands
- accelerator executes sequentially
- host can pipeline by double-buffering (future)

Back-pressure
Host must not overflow accelerator buffers.
We enforce:
- bounded queue in host (see host_tools_async)
- ack/response per command or per batch

Determinism
Commands execute in-order unless explicitly stated otherwise.

Why this is realistic
Many NPUs expose command streams (even if not JSON).
We are writing the spec first so our FPGA kernels can fit cleanly later.
