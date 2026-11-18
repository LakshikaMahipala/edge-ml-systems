# Platform diagram 
```mermaid
flowchart LR
  subgraph Host["Host (Python/C++)"]
    A["App / Bench scripts"] --> B["Scheduler / Queue<br/>(host_tools_async)"]
    B --> C["Protocol (frames + req_id)<br/>UART now"]
  end

  subgraph FPGA["FPGA Backend"]
    D["RX/TX + Frame decode"] --> E["Op Dispatcher"]
    E --> F["INT8_FC kernel"]
    E --> G["INT8_DWCONV1D kernel"]
    F --> H["OUTPUT buffer"]
    G --> H
  end

  subgraph Micro["Micro runtime mindset (TinyML)"]
    M1["TFLite Micro-style skeleton"] --> M2["Op set constraint"]
    M2 --> M3["MAC/cycle proxy cost model"]
  end

  C --> D
  H --> D
  D --> C

  subgraph Spec["Platform specs"]
    S1["micro_npu_platform (command + buffers)"]
    S2["fpga_platform_spec (UART + FPGA ops)"]
  end

  S1 -.defines.-> B
  S2 -.defines.-> E
  Micro -.informs op set.-> S1

Commit: `Add Week 8 platform diagram (Mermaid)`

---

# 2) `docs/api_sketch.md` (the actual “contract in human terms”)

Copy-paste:

```md
# API sketch (Week 8)

This is the minimal API contract that ties host ↔ FPGA ↔ specs.

## Transport
- UART framed packets with CRC
- Request/response matching uses `req_id` (1 byte)

## Message forms (v0 Mode A: direct op call)

### FC call
Host → FPGA payload:
- [req_id][OP_ID=FC][IN][OUT][SHIFT][X bytes][(optional) weights selector]

FPGA → Host payload:
- [req_id][Y bytes]

### DWConv1D call
Host → FPGA payload:
- [req_id][OP_ID=DWCONV1D][C][L][SHIFT][X bytes]

FPGA → Host payload:
- [req_id][Y bytes]

## Long-term target (Mode B: buffered command stream)
- host writes INPUT/WEIGHTS buffers
- host sends EXEC command with op + addresses
- FPGA runs and writes OUTPUT buffer
This matches `micro_npu_platform/spec/04_command_stream_and_scheduler.md`

## Determinism rule
Given identical bytes, output must match bit-exactly with the reference Python model.
