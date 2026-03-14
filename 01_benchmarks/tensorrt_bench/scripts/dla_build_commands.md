DLA build/run command cookbook 

Note
Exact flags may vary by TensorRT version / platform, but the workflow is stable:
- target DLA
- optionally allow GPU fallback
- optionally choose precision

A) DLA + GPU fallback (first attempt)
trtexec --onnx results/model.onnx \
  --saveEngine results/model_dla_fp16.plan \
  --fp16 \
  --useDLACore=0 \
  --allowGPUFallback \
  --verbose

B) DLA-only (strict)
trtexec --onnx results/model.onnx \
  --saveEngine results/model_dla_only_fp16.plan \
  --fp16 \
  --useDLACore=0 \
  --verbose

C) INT8 on DLA (requires calibration cache)
trtexec --onnx results/model.onnx \
  --saveEngine results/model_dla_int8.plan \
  --int8 \
  --useDLACore=0 \
  --allowGPUFallback \
  --calib=results/calib/resnet18.cache \
  --verbose

What to look for in logs
- DLA layer assignment / fallbacks
- warnings about unsupported ops or parameter limits
- engine build success/failure

References
- TensorRT DLA documentation (“Working with DLA”)
