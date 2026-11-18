# Examples

Example pipeline DAG:
preprocess → infer → postprocess

Heterogeneous version:
preprocess on CPU
infer on GPU or FPGA
postprocess on CPU

We will encode these as JSON DAGs and run HEFT.
