FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake llvm-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Core ML deps
RUN pip install --no-cache-dir numpy torch torchvision

# TVM Python package (works in many environments; if it fails, use source build later)
RUN pip install --no-cache-dir apache-tvm

WORKDIR /work
