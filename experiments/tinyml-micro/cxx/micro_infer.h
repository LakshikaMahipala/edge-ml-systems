#pragma once
#include <cstdint>

struct InferResult {
  int top1;
  float conf;
};

class MicroInfer {
public:
  MicroInfer();
  bool init();                  // load model + allocate arena (later)
  InferResult run(const int8_t* input, int input_bytes); // invoke (later)
};
