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
