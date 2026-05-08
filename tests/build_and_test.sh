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
# Requirements:
#   - CUDA toolkit with nvcc (sm_90a support required)
#   - Python with torch, sglang (sglang does NOT need to be fully installed;
#     the pool/backend modules just need to be importable)
#   - CUTLASS headers (see ../install.sh for options)
#
# Usage:
#   cd tests/
#   bash build_and_test.sh
#
# Environment variables (optional):
#   CUTLASS_INCLUDE_DIR   path to CUTLASS include/ (auto-detected if not set)
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS_DIR="$REPO/tests"
CSRC_DIR="$REPO/csrc"

echo "=== DirectKV Correctness Tests ==="
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
echo "      Compiles and runs a standalone SM90a kernel correctness check."

cd "$TESTS_DIR"
rm -f test_directkv_cpu_kv

# Locate CUTLASS headers
if [ -n "${CUTLASS_INCLUDE_DIR:-}" ]; then
    CUTLASS_INC="$CUTLASS_INCLUDE_DIR"
elif python3 -c "import deep_gemm" 2>/dev/null; then
    CUTLASS_INC=$(python3 -c "
import deep_gemm, os
print(os.path.join(os.path.dirname(deep_gemm.__file__), 'include'))
")
else
    echo "[ERROR] CUTLASS headers not found."
    echo "  Option A: pip install deep-gemm"
    echo "  Option B: export CUTLASS_INCLUDE_DIR=/path/to/cutlass/include"
    exit 1
fi
echo "  CUTLASS: $CUTLASS_INC"

nvcc -O2 -arch=sm_90a -std=c++17 --expt-relaxed-constexpr \
    -I"$CUTLASS_INC" \
    -I"$CSRC_DIR" \
    "$TESTS_DIR/test_directkv_cpu_kv.cu" \
    -o "$TESTS_DIR/test_directkv_cpu_kv" \
    -lcuda

echo "  Build: OK"
"$TESTS_DIR/test_directkv_cpu_kv"
echo ""
echo "=== All tests PASSED ==="
