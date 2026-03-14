Run Checklist (when GPU machine is available)

Before running
- Confirm GPU + driver installed: nvidia-smi
- Confirm TensorRT installed:
  - trtexec exists on PATH
- Confirm Python deps:
  pip install -r tensorrt_bench/requirements.txt

TensorRT run order
1) Export ONNX:
   python tensorrt_bench/scripts/export_onnx.py ...

2) Build engines:
   - FP32
   - FP16
   - INT8 (requires calibration tensor)

3) Benchmark:
   python tensorrt_bench/scripts/run_trt_benchmark.py ... --run --log ...

4) Parse logs:
   python tensorrt_bench/scripts/parse_trtexec_log.py ...

5) Update tables:
   docs/mini_project_3_trt_vs_ort_vs_pytorch.md

FPGA simulation
- Install iverilog
- Run UART echo sim script
- Run fixed_point_lib sim script
- Save terminal output and waveform screenshots later (optional)

What to commit
- Do NOT commit huge binary engines.
- Commit JSON summaries + your report tables + screenshots if used.
