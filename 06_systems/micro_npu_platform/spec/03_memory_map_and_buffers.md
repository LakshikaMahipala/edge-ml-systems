Memory map and buffers (v0)

We define a simple addressable buffer model to support host<->accelerator interaction.

Buffers (conceptual)
- INPUT buffer
- OUTPUT buffer
- WEIGHT buffer
- SCRATCH buffer (optional)

v0 memory map (logical)
INPUT   : base=0x0000 size = input_bytes
OUTPUT  : base=0x1000 size = output_bytes
WEIGHTS : base=0x2000 size = weights_bytes
SCRATCH : base=0x3000 size = scratch_bytes

Transport mapping
- UART protocol frames write/read these buffers.
- Later PCIe/Ethernet can map to the same logical addresses.

Why this matters
A stable memory map allows:
- reproducible integration
- a compiler to emit addresses
- a runtime to schedule without hardcoding shapes
