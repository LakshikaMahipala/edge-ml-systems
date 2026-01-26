Docker Quickstart

Goal
Provide a reproducible environment to build:
- cpp_preproc (C++)
- cpp_preproc_ext (pybind11)
and run inference_bench scripts.

Build image (later)
From repo root:
- docker build -f docker/Dockerfile -t edge-ml-systems:dev .

Run interactive shell
- docker run --rm -it edge-ml-systems:dev bash

Verify extension import
Inside container:
- python -c "import cpp_preproc_ext; print('cpp_preproc_ext OK')"

Run benchmarks (examples)
- python inference_bench/run_pytorch_benchmark.py --model resnet18 --device cpu
- python inference_bench/run_preproc_speedup.py --model resnet18 --device cpu --iters 50 --warmup 10 --image /workspace/data/sample.jpg
- python inference_bench/run_file_io_benchmark.py --path /workspace/data/large.bin --iters 20

Mount local data folder
If you want to use local files:
- docker run --rm -it -v $(pwd)/data:/workspace/data edge-ml-systems:dev bash

Notes
- This image is a dev container (build + run).
- Performance numbers inside containers may differ slightly from host due to isolation and filesystem differences.

Docker (reproducible dev environment)
- Build: docker build -f docker/Dockerfile -t edge-ml-systems:dev .
- Run:   docker run --rm -it edge-ml-systems:dev bash
- Verify: python -c "import cpp_preproc_ext; print('OK')"
See docker/README.md for full instructions.
