#!/usr/bin/env bash
# Build the cuBLAS fp16 GEMV baseline driver. cuBLAS ships with the CUDA toolkit;
# no torch. The driver is gitignored (a binary); this is the reproducible build.
set -euo pipefail
cd "$(dirname "$0")"

# No __global__ kernels here, so plain g++ with the CUDA headers/libs works
# (avoids depending on nvcc being on PATH).
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

g++ -O2 -std=c++17 cublas_gemv.cpp -o cublas_gemv \
    -I "$CUDA_INC" -L "$CUDA_LIB" -lcublas -lcudart \
    -Wl,-rpath,"$CUDA_LIB"

echo "built: $(pwd)/cublas_gemv"
