TFLite Micro quickstart (conceptual)

What TFLite Micro is
A minimal inference runtime designed for microcontrollers:
- no dynamic memory allocation (typically)
- small operator resolver
- fixed tensor arena for all buffers
- runs .tflite models (quantized INT8 commonly)

Execution model (high level)
1) Map model (flatbuffer) from flash into memory
2) Create operator resolver (only ops you need)
3) Allocate a tensor arena (a static byte buffer)
4) Invoke the interpreter repeatedly on new inputs

What we build today
- a C++ project skeleton that mirrors TFLM structure
- a placeholder model_data.h (real model bytes will be added later)
- a cost estimator script for “cycle-ish” reasoning
