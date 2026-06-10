#!/bin/bash

set -euo pipefail

mkdir -p build

CUDA_SOURCE="${CUDA_SOURCE:-cuda_versions/cuda_v0.cu}"
CUDA_BINARY="${CUDA_BINARY:-build/cuda_v0}"

nvcc -O2 -std=c++17 "${CUDA_SOURCE}" -o "${CUDA_BINARY}"

python benchmark.py \
	--image Broadway_tower_edit.jpg \
	--seams 10 \
	--runs 2 \
	--warmup 0 \
	--progress \
	--cuda-source "${CUDA_SOURCE}" \
	--cuda-binary "${CUDA_BINARY}" \
	--skip-cpu
