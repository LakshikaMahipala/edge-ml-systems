# docker/Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# -------------------------
# System deps
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git curl ca-certificates \
    build-essential cmake pkg-config \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Make "python" available
RUN ln -sf /usr/bin/python3 /usr/bin/python

# -------------------------
# Python deps (inference_bench)
# -------------------------
# Copy requirements first for better docker layer caching
COPY inference_bench/requirements.txt /workspace/inference_bench/requirements.txt

RUN pip3 install --no-cache-dir -r /workspace/inference_bench/requirements.txt

# -------------------------
# Copy repo content
# -------------------------
COPY . /workspace

# -------------------------
# Build C++ + pybind11 extension
# -------------------------
# Build in /workspace/build_cpp so the compiled .so is in a known place.
RUN cmake -S /workspace/cpp_preproc -B /workspace/build_cpp -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /workspace/build_cpp -j

# Ensure Python can import the extension from build folder
ENV PYTHONPATH=/workspace/build_cpp:${PYTHONPATH}

# Default command
CMD ["bash"]
