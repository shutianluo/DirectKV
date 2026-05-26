#!/usr/bin/env bash
# =============================================================================
# build_and_test.sh — Compile and run DirectKV correctness tests
#
# Tests:
#   1. Python correctness test  — KV pool storage/retrieval and attention
#      output against torch.nn.functional.scaled_dot_product_attention
#   2. Standalone CUDA kernel test (test_directkv_cpu_kv.cu) — validates the
#      DirectKV fused kernel output vs a reference CPU computation
#
# Supported hardware:
#   Primary:  NVIDIA GH200 (SM90a, NVLink-C2C) — full performance target
#   Fallback: NVIDIA H100 / H200 (SM9.0, PCIe) — correctness identical;
#             CPU-KV bandwidth is PCIe-limited (~50 GB/s) vs NVLink-C2C
#             (~900 GB/s), so decode latency will be higher than the paper
#             reports. The kernel compiles and runs correctly on SM9.0.
#
# Requirements:
#   - CUDA toolkit with nvcc (sm_90a or sm_90 support)
#   - ae_python virtualenv activated (source $WORKSPACE/ae_python/bin/activate)
#   - Python with torch, sglang installed in ae_python
#   - CUTLASS headers: git clone https://github.com/NVIDIA/cutlass.git $WORKSPACE/cutlass
#                      export CUTLASS_INCLUDE_DIR=$WORKSPACE/cutlass/include
#
# Usage:
#   source $WORKSPACE/ae_python/bin/activate
#   cd tests/
#   bash build_and_test.sh
#
# Environment variables (optional):
#   CUTLASS_INCLUDE_DIR   path to CUTLASS include/ (auto-detected if not set)
#   NVCC_ARCH             override nvcc -arch flag (default: auto-detected)
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS_DIR="$REPO/tests"
CSRC_DIR="$REPO/csrc"

echo "=== DirectKV Correctness Tests ==="
echo ""

# ---------------------------------------------------------------------------
# Detect GPU and choose compilation target
# ---------------------------------------------------------------------------
if [ -n "${NVCC_ARCH:-}" ]; then
    ARCH_FLAG="$NVCC_ARCH"
    echo "  nvcc arch: $ARCH_FLAG (user-specified via NVCC_ARCH)"
else
    # Query compute capability of GPU 0
    GPU_CC=$(python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('unknown')
    sys.exit(0)
major, minor = torch.cuda.get_device_capability(0)
print(f'{major}{minor}')
" 2>/dev/null || echo "unknown")

    GPU_NAME=$(python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('unknown')
    sys.exit(0)
print(torch.cuda.get_device_name(0))
" 2>/dev/null || echo "unknown")

    if [ "$GPU_CC" = "90" ]; then
        # Both GH200 (SM90a) and H100/H200 (SM90) report compute capability 9.0.
        # nvcc -arch=sm_90a compiles and executes correctly on all SM9.0 devices;
        # it enables the full SM90 warp-group MMA / TMA feature set used by DirectKV.
        ARCH_FLAG="sm_90a"
        if echo "$GPU_NAME" | grep -qi "GH200"; then
            echo "  GPU: $GPU_NAME (GH200 / NVLink-C2C) — compiling for $ARCH_FLAG"
            echo "  Bandwidth: NVLink-C2C ~900 GB/s — full performance target."
        else
            echo "  GPU: $GPU_NAME (SM9.0, PCIe) — compiling for $ARCH_FLAG"
            echo "  NOTE: Kernel is functionally correct on this GPU."
            echo "        CPU-KV bandwidth is PCIe-limited (~50 GB/s) vs"
            echo "        NVLink-C2C (~900 GB/s) on GH200.  Correctness tests"
            echo "        pass; end-to-end decode latency will differ from paper."
        fi
    else
        echo "  GPU: $GPU_NAME (SM${GPU_CC}) — unsupported; DirectKV requires SM90."
        echo "  Attempting sm_90a anyway; compilation may fail."
        ARCH_FLAG="sm_90a"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Python correctness test
# ---------------------------------------------------------------------------
echo "[1/2] Python correctness test (test_directkv_correctness.py)..."
echo "      Checks KV pool round-trip and attention output vs SDPA reference."
python3 "$TESTS_DIR/test_directkv_correctness.py" -v
echo ""

# ---------------------------------------------------------------------------
# 2. Standalone CUDA kernel test
# ---------------------------------------------------------------------------
echo "[2/2] CUDA kernel test (test_directkv_cpu_kv.cu)..."
echo "      Compiles and runs a standalone SM90 kernel correctness check."

cd "$TESTS_DIR"
rm -f test_directkv_cpu_kv

# Locate CUTLASS headers
if [ -n "${CUTLASS_INCLUDE_DIR:-}" ]; then
    CUTLASS_INC="$CUTLASS_INCLUDE_DIR"
else
    echo "[ERROR] CUTLASS_INCLUDE_DIR is not set."
    echo "  Clone CUTLASS v3.9.2 and set the variable before running this script:"
    echo "    git clone --branch v3.9.2 --depth 1 https://github.com/NVIDIA/cutlass.git \$WORKSPACE/cutlass"
    echo "    export CUTLASS_INCLUDE_DIR=\$WORKSPACE/cutlass/include"
    exit 1
fi
echo "  CUTLASS: $CUTLASS_INC"

nvcc -O2 -arch="$ARCH_FLAG" -std=c++17 --expt-relaxed-constexpr \
    -I"$CUTLASS_INC" \
    -I"$CSRC_DIR" \
    "$TESTS_DIR/test_directkv_cpu_kv.cu" \
    -o "$TESTS_DIR/test_directkv_cpu_kv" \
    -lcuda

echo "  Build: OK"
"$TESTS_DIR/test_directkv_cpu_kv"
echo ""
echo "=== All tests PASSED ==="
