#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "preproc.hpp"

namespace py = pybind11;

static py::array_t<float> tensor_to_numpy(const preproc::Tensor& t) {
  // Create numpy array with shape (N, C, H, W)
  std::vector<ssize_t> shape = {t.n, t.c, t.h, t.w};
  std::vector<ssize_t> strides = {
    static_cast<ssize_t>(t.c * t.h * t.w * sizeof(float)),
    static_cast<ssize_t>(t.h * t.w * sizeof(float)),
    static_cast<ssize_t>(t.w * sizeof(float)),
    static_cast<ssize_t>(sizeof(float))
  };

  // py::array_t makes a copy by default if we pass pointer without capsule.
  // To keep correctness and simplicity for v0, we return a new owned array and copy data into it.
  py::array_t<float> arr(shape, strides);
  std::memcpy(arr.mutable_data(), t.data.data(), t.data.size() * sizeof(float));
  return arr;
}

PYBIND11_MODULE(cpp_preproc_ext, m) {
  m.doc() = "C++ preprocessing extension: load/resize/normalize -> NCHW float32";

  py::class_<preproc::NormalizeParams>(m, "NormalizeParams")
      .def(py::init<>())
      .def_readwrite("mean", &preproc::NormalizeParams::mean)
      .def_readwrite("std", &preproc::NormalizeParams::std);

  m.def(
      "load_resize_normalize",
      [](const std::string& image_path, int out_h, int out_w) {
        preproc::Tensor t = preproc::load_resize_normalize(image_path, out_h, out_w);
        return tensor_to_numpy(t);
      },
      py::arg("image_path"),
      py::arg("out_h") = 224,
      py::arg("out_w") = 224
  );

  m.def(
      "tensor_stats",
      [](py::array_t<float> arr) {
        // Accept numpy array and compute stats in Python later if needed.
        // Keeping this minimal for Day 4.
        auto buf = arr.request();
        const float* data = static_cast<const float*>(buf.ptr);
        size_t n = 1;
        for (auto s : buf.shape) n *= static_cast<size_t>(s);

        float mn = data[0], mx = data[0];
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
          float v = data[i];
          if (v < mn) mn = v;
          if (v > mx) mx = v;
          sum += v;
        }
        float mean = static_cast<float>(sum / static_cast<double>(n));
        return py::dict("min"_a=mn, "max"_a=mx, "mean"_a=mean, "n"_a=n);
      }
  );
}
