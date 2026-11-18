#include "preproc.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: preproc_cli <image_path> [H] [W]\n";
    return 1;
  }

  std::string path = argv[1];
  int H = (argc >= 3) ? std::stoi(argv[2]) : 224;
  int W = (argc >= 4) ? std::stoi(argv[3]) : 224;

  try {
    preproc::Tensor t = preproc::load_resize_normalize(path, H, W);
    preproc::Stats s = preproc::tensor_stats(t);

    std::cout << "OK\n";
    std::cout << "Tensor shape: N=" << t.n << " C=" << t.c << " H=" << t.h << " W=" << t.w << "\n";
    std::cout << "Stats: min=" << s.min_val << " max=" << s.max_val << " mean=" << s.mean_val << "\n";
    std::cout << "First 10 values: ";
    for (int i = 0; i < 10 && i < (int)t.data.size(); ++i) {
      std::cout << t.data[i] << (i == 9 ? "" : ", ");
    }
    std::cout << "\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }

  return 0;
} 
