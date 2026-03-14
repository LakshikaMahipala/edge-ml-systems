Python Extension (pybind11) 

Module name
- cpp_preproc_ext

What it exposes
- load_resize_normalize(image_path, out_h, out_w) -> numpy float32 array (N,C,H,W)
- tensor_stats(numpy_array) -> dict(min/max/mean)

Build (later, locally)
Prereqs:
- Python dev environment
- CMake + C++ compiler
- OpenCV dev package

Commands:
- mkdir -p build && cd build
- cmake -S ../cpp_preproc -B . -DCMAKE_BUILD_TYPE=Release     
- cmake --build . -j

Where is the .so / .pyd?
- In the build folder (name depends on OS)
Example:
- cpp_preproc_ext.cpython-311-x86_64-linux-gnu.so

How to import (from repo root)
Option A: add build folder to PYTHONPATH
- export PYTHONPATH=$PYTHONPATH:$(pwd)/build
- python -c "import cpp_preproc_ext; print(cpp_preproc_ext.__doc__)"

Option B: copy the built module into inference_bench/ (not recommended long-term)
