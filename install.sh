#!/usr/bin/env bash
# =============================================================================
# install.sh — Install DirectKV backend into a working SGLang installation
#
# DirectKV registers itself into SGLang via two files placed in site-packages:
#   directkv_startup.pth        — tells Python to import the hook at startup
#   directkv_startup_hook.py    — patches SGLang's attention backend registry
#
# Additionally, the backend module (directkv_backend.py) and pool module
# (directkv_pool.py) are placed where SGLang can import them.
#
# Prerequisites:
#   pip install sglang[all]==0.4.9.post6
#   pip install pybind11 numpy transformers
#
# CUTLASS headers (required to compile the CUDA kernel):
#   Option A (recommended): pip install deep-gemm
#     → CUTLASS headers land in $(python -c "import deep_gemm; print(deep_gemm.__file__.rsplit('/',1)[0])")/include
#   Option B: clone CUTLASS and set CUTLASS_INCLUDE_DIR=/path/to/cutlass/include
#     git clone https://github.com/NVIDIA/cutlass.git
#     export CUTLASS_INCLUDE_DIR=/path/to/cutlass/include
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE=$(python3 -c "import site; print(site.getusersitepackages())")
mkdir -p "$SITE"

echo "[1/4] Copying startup hook..."
cp "$REPO/directkv/directkv_startup.pth"    "$SITE/"
cp "$REPO/directkv/directkv_startup_hook.py" "$SITE/"

echo "[2/4] Copying backend module into SGLang..."
mkdir -p "$SITE/sglang/srt/layers/attention/directkv_kernel"
cp "$REPO/directkv/directkv_backend.py"      "$SITE/sglang/srt/layers/attention/"
cp "$REPO/directkv/directkv_kernel/"*.cu      "$SITE/sglang/srt/layers/attention/directkv_kernel/"
cp "$REPO/directkv/directkv_kernel/__init__.py" "$SITE/sglang/srt/layers/attention/directkv_kernel/"

echo "[3/4] Copying pool module into SGLang..."
mkdir -p "$SITE/sglang/srt/mem_cache"
cp "$REPO/directkv/directkv_pool.py"         "$SITE/sglang/srt/mem_cache/"
cp "$REPO/directkv/directkv_request_pool.py" "$SITE/sglang/srt/mem_cache/"

echo "[4/4] Verifying..."
python3 - <<'PYEOF'
import sys
try:
    import directkv_startup_hook  # noqa: F401
    print("  directkv_startup_hook: OK")
except Exception as e:
    print(f"  directkv_startup_hook: WARN ({e})")
try:
    from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool
    print("  DirectKVTokenToKVPool: OK")
except Exception as e:
    print(f"  DirectKVTokenToKVPool: WARN ({e})")
PYEOF

echo ""
echo "Installation complete. Use --attention-backend directkv when launching SGLang."
echo ""
echo "The CUDA kernel is JIT-compiled on first use. Ensure CUTLASS headers are"
echo "available (see prerequisites at the top of this script)."
