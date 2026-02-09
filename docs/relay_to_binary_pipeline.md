# Relay → schedule → binary 

## The big picture

Relay (Graph IR) decides **what** operations run and in what dataflow order.  
Lower-level IR/schedules decide **how** each operation is implemented.

```mermaid
flowchart LR
  A["PyTorch / ONNX model"] --> B["Relay (Graph IR)"]
  B --> C["Relay passes<br/>- op counting<br/>- fusion<br/>- layout transforms"]
  C --> D["Lowering to kernel IR (TIR)"]
  D --> E["Scheduling / Autotuning<br/>tiling, unroll, vectorize"]
  E --> F["Codegen"]
  F --> G["Binary / runtime"]
