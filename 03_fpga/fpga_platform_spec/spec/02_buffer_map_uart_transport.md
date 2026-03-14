Buffer map + UART transport

Logical memory map (same as micro_npu_platform)
INPUT   : 0x0000
OUTPUT  : 0x1000
WEIGHTS : 0x2000
SCRATCH : 0x3000

UART framing rules (current)
Frame types:
- TYPE_INPUT: host->fpga input payload
- TYPE_OUTPUT: fpga->host output payload
- TYPE_PING/PONG

Request/response matching
We include req_id (1 byte) as the first byte of payload.
- INPUT payload:  [req_id][command_or_data...]
- OUTPUT payload: [req_id][result_data...]

Two transport modes (v0)
Mode A: “direct op call” (simple)
- payload = [req_id][OP_ID][params...][input bytes...]
- FPGA computes and replies with output bytes.

Mode B: “buffered command” (aligns with command stream)
- payload writes buffers, then sends EXEC command.
We will migrate to Mode B later.

v0 choice for week 8
Use Mode A for fastest progress and to validate async host driver.

Stability requirements
- bounded host queues (already implemented)
- timeout and retry policy (host side)
- CRC errors counted
