#include <iostream>
#include <vector>
#include "micro_infer.h"

int main() {
  MicroInfer inf;
  if (!inf.init()) {
    std::cerr << "Init failed\n";
    return 1;
  }

  // Dummy input for now
  std::vector<int8_t> x(28 * 28, 0);
  auto out = inf.run(x.data(), (int)x.size());

  std::cout << "Dummy run: top1=" << out.top1 << " conf=" << out.conf << "\n";
  return 0;
}
