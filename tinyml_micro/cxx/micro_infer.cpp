#include "micro_infer.h"

MicroInfer::MicroInfer() {}

bool MicroInfer::init() {
  // Placeholder for TFLite Micro init:
  // - load g_model flatbuffer
  // - build op resolver
  // - allocate tensor arena
  return true;
}

InferResult MicroInfer::run(const int8_t* /*input*/, int /*input_bytes*/) {
  // Placeholder:
  // - copy input to model input tensor
  // - interpreter.Invoke()
  // - read output tensor and return top1/conf
  return InferResult{0, 0.0f};
}
