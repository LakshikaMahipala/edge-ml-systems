Edge Build & Deployment 

Week 3 scope (what the repo supports)
- File I/O benchmark (p50/p99 + throughput)
- Cross-compilation toolchain skeleton (ARM64/ARMv7)
- Docker reproducible build/run environment
- Transfer/copy-cost budget model (PCIe fundamentals)
- Camera/video decode benchmark
- Mini-project 1B: ONNX Runtime vs PyTorch latency comparison

Commands (run later)
1) File I/O
python inference_bench/run_file_io_benchmark.py --path path/to/large.bin --iters 30 --chunk_kb 256

2) Cross-compile (skeleton)
cmake -S cpp_preproc -B build_aarch64 -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux-gnu.cmake
cmake --build build_aarch64 -j

3) Docker
docker build -f docker/Dockerfile -t edge-ml-systems:dev .
docker run --rm -it edge-ml-systems:dev bash

4) Transfer budget
python inference_bench/run_transfer_budget_demo.py --n 1 --c 3 --h 224 --w 224 --dtype_bytes 4

5) Video decode
python inference_bench/run_video_decode_benchmark.py --video path/to/video.mp4 --warmup 20 --frames 200

6) PyTorch baseline
python inference_bench/run_pytorch_benchmark.py --model resnet18 --device cpu --input_size 224 --warmup 20 --iters 100 --topk 5 --save_json

7) ONNX Runtime baseline
python inference_bench/run_onnxruntime_benchmark.py --model resnet18 --input_size 224 --batch 1 --warmup 20 --iters 100 --save_json

Results ledger
- docs/metrics.md is the single source of truth for reported numbers.
- Commit only representative JSON outputs (avoid spam).
