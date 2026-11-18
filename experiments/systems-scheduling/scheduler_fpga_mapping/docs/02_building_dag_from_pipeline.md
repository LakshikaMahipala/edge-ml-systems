# Building a DAG from an inference pipeline

Pipeline tasks:
- decode/preprocess
- infer_block1
- infer_block2
- postprocess

Edges carry activation tensors:
- bytes moved determines comm cost if tasks are on different devices.

We encode:
- per-task per-device costs from latency registry
- edge bytes from tensor shapes

Then we run HEFT to decide placements and start/end times.
