#!/usr/bin/env bash
# build_arm.sh — ARM/aarch64 build for pacpu (no ISPC required)
#
# Usage:  bash build_arm.sh <model_name> <tp_degree>
# e.g.    bash build_arm.sh llama3_8b 1
#
# Supported models: llama3_8b  llama2_7b  llama2_13b  llama2_70b  llama3_70b

set -euo pipefail

MODEL="${1:?Usage: bash build_arm.sh <model_name> <tp_degree>}"
TP="${2:?Usage: bash build_arm.sh <model_name> <tp_degree>}"

TORCH_DIR=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/Torch
CXX=$(which g++)
PACPU_SRC="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PACPU_SRC}/build_arm"
STAGE_DIR="${PACPU_SRC}/build_arm_stage"

echo "=== pacpu ARM build ==="
echo "  model=${MODEL}  tp=${TP}"
echo "  compiler: ${CXX} ($(${CXX} --version | head -1))"
echo "  arch: $(uname -m)"
echo "  torch: ${TORCH_DIR}"

# Stage source files with CMakeLists_arm.txt as CMakeLists.txt
mkdir -p "${STAGE_DIR}"
cp "${PACPU_SRC}"/*.h "${PACPU_SRC}"/*.cpp "${STAGE_DIR}/"
cp "${PACPU_SRC}/CMakeLists_arm.txt" "${STAGE_DIR}/CMakeLists.txt"

# Configure
mkdir -p "${BUILD_DIR}"
cmake \
    -B "${BUILD_DIR}" \
    -S "${STAGE_DIR}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTorch_DIR="${TORCH_DIR}" \
    -DModel="${MODEL}" \
    -DTP="${TP}"

# Build
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

LIB="${BUILD_DIR}/libpacpu-${MODEL}-tp${TP}.so"
if [ ! -f "${LIB}" ]; then
    echo "ERROR: expected ${LIB} not found" >&2
    exit 1
fi

mkdir -p "${PACPU_SRC}/build"
cp "${LIB}" "${PACPU_SRC}/build/"

echo ""
echo "=== Build successful ==="
echo "  ${PACPU_SRC}/build/libpacpu-${MODEL}-tp${TP}.so  ($(du -sh "${PACPU_SRC}/build/libpacpu-${MODEL}-tp${TP}.so" | cut -f1))"
echo ""
echo "Next: update evaluation/configs/*.json:"
echo "  \"library\": \"libpacpu-${MODEL}-tp${TP}.so\""
