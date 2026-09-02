#!/usr/bin/env bash
# Build the ggml quantized-GEMV baseline driver (Q4_0/Q8_0/Q4_K) against the
# llama.cpp checkout in ./src (pinned in ./PIN). Run from this directory after
# src/ has been built with CUDA (see the `baselines` skill for the llama.cpp
# cmake line: -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89").
#
# The driver is gitignored (a binary); this script is the reproducible build step.
set -euo pipefail
cd "$(dirname "$0")"

GGML_INC=src/ggml/include
GGML_LIB=src/build/bin
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

g++ -O2 -std=c++17 gemv_bench.cpp -o gemv_bench \
    -I "$GGML_INC" -I "$CUDA_INC" \
    -L "$GGML_LIB" -lggml -lggml-base -lggml-cpu -lggml-cuda \
    -L "$CUDA_LIB" -lcudart \
    -Wl,-rpath,"\$ORIGIN/$GGML_LIB" -Wl,-rpath,"$CUDA_LIB"

echo "built: $(pwd)/gemv_bench"

# fp32 -> GGUF-block quantizer (for formats gguf-py cannot quantize, e.g. Q4_K).
# CPU-only (ggml_quantize_chunk); no CUDA needed but links the same libs.
g++ -O2 -std=c++17 quantize_weight.cpp -o quantize_weight \
    -I "$GGML_INC" \
    -L "$GGML_LIB" -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,"\$ORIGIN/$GGML_LIB"

echo "built: $(pwd)/quantize_weight"
