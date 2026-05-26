#!/usr/bin/env bash
# =============================================================================
# install.sh — Install DirectKV backend into a working SGLang installation
#
# DirectKV registers itself into SGLang via two files placed in site-packages:
#   directkv_startup.pth        — tells Python to import the hook at startup
#   directkv_startup_hook.py    — patches SGLang's attention backend registry
#
# Additionally the following modules are placed where SGLang can import them:
#   directkv_backend.py / directkv_smpv2_backend.py — attention backends
#   directkv_pool.py / directkv_request_pool.py     — CPU-pinned KV pool
#   baseline/pie/kv_cache_offload.py                 — Pie KV offload module
#   opt_model.py → sglang/srt/models/opt.py         — OPT model for SGLang
#
# SGLang 0.4.9.post6 compatibility patches (applied automatically):
#   server_args.py  — adds "directkv" / "directkv-smpv2" to --attention-backend
#   sgl_kernel      — replaces broken __init__.py with an ABI-safe stub
#   torchao_utils   — adds early-return guard to avoid crash on empty config
#
# Prerequisites:
#   Activate the ae_python virtualenv before running this script:
#     source /path/to/ae_python/bin/activate
#   pip install sglang[all]==0.4.9.post6
#   pip install pybind11 numpy transformers
#
# CUTLASS headers (required to compile the CUDA kernel):
#   Clone CUTLASS v3.9.2 (later versions changed SM100 APIs and will fail):
#     git clone --branch v3.9.2 --depth 1 https://github.com/NVIDIA/cutlass.git $WORKSPACE/cutlass
#     export CUTLASS_INCLUDE_DIR=$WORKSPACE/cutlass/include
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE=$(python3 -c "import sys, site; print(site.getsitepackages()[0] if sys.prefix != sys.base_prefix else site.getusersitepackages())")
mkdir -p "$SITE"

echo "[0/4] Creating helper .pth files for Neo and FlexGen..."
# neo/csrc contains the compiled swiftllm_c extension (not installed by pip install -e neo/)
NEO_CSRC="$REPO/baseline/neo/csrc"
if [ -d "$NEO_CSRC" ]; then
    echo "$NEO_CSRC" > "$SITE/neo_csrc.pth"
    echo "  neo_csrc.pth → $NEO_CSRC"
else
    echo "  WARN: $NEO_CSRC not found — skipping neo_csrc.pth"
fi

echo "[1/4] Copying startup hook and env..."
cp "$REPO/directkv/directkv_startup.pth"    "$SITE/"
cp "$REPO/directkv/directkv_startup_hook.py" "$SITE/"

# Create env pth: set DIRECTKV_SMPV2_KERNEL_SRC and CUTLASS_INCLUDE_DIR defaults
CUTLASS_GUESS=$(python3 -c "
import os, site
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    p = os.path.join(sp, 'flashinfer', 'data', 'cutlass', 'include')
    if os.path.isdir(p):
        print(p); break
" 2>/dev/null)

cat > "$SITE/directkv_env.pth" << PTHEOF
import os; os.environ.setdefault('DIRECTKV_SMPV2_KERNEL_SRC', '$REPO/kernel'); os.environ.setdefault('CUTLASS_INCLUDE_DIR', '${CUTLASS_GUESS:-}')
PTHEOF
echo "  directkv_env.pth written (SMPV2_KERNEL_SRC=$REPO/kernel)"

echo "[2/4] Copying backend module into SGLang..."
mkdir -p "$SITE/sglang/srt/layers/attention/directkv_kernel"
cp "$REPO/directkv/directkv_backend.py"        "$SITE/sglang/srt/layers/attention/"
cp "$REPO/directkv/directkv_smpv2_backend.py"  "$SITE/sglang/srt/layers/attention/"
cp "$REPO/directkv/kernel/"*.cu                "$SITE/sglang/srt/layers/attention/directkv_kernel/"
cp "$REPO/directkv/kernel/__init__.py"         "$SITE/sglang/srt/layers/attention/directkv_kernel/"

# Ensure smpv2_extension.cu is present (required by directkv-smpv2 backend)
if [ ! -f "$SITE/sglang/srt/layers/attention/directkv_kernel/smpv2_extension.cu" ] && \
   [ -f "$REPO/directkv/kernel/smpv2_extension.cu" ]; then
    cp "$REPO/directkv/kernel/smpv2_extension.cu" \
       "$SITE/sglang/srt/layers/attention/directkv_kernel/"
fi

echo "[3/4] Copying pool module into SGLang..."
mkdir -p "$SITE/sglang/srt/mem_cache"
cp "$REPO/directkv/directkv_pool.py"         "$SITE/sglang/srt/mem_cache/"
cp "$REPO/directkv/directkv_request_pool.py" "$SITE/sglang/srt/layers/attention/"
cp "$REPO/baseline/pie/kv_cache_offload.py"   "$SITE/sglang/srt/layers/"
cp "$REPO/directkv/opt_model.py"             "$SITE/sglang/srt/models/opt.py"

echo "[3b/4] Patching sglang server_args to add directkv to --attention-backend choices..."
SERVER_ARGS="$SITE/sglang/srt/server_args.py"
if [ -f "$SERVER_ARGS" ]; then
    python3 - <<'PYEOF' "$SERVER_ARGS"
import pathlib, sys
p = pathlib.Path(sys.argv[1])
txt = p.read_text()
marker = '"aiter",'
if '"directkv"' not in txt:
    new_block = '"aiter",\n                "directkv",\n                "directkv-smpv2",'
    txt2 = txt.replace(marker, new_block, 1)
    if txt2 == txt:
        print(f"  WARN: could not find insertion point in {p}")
    else:
        p.write_text(txt2)
        print("  server_args.py patched: directkv / directkv-smpv2 added to choices")
else:
    print("  server_args.py already patched")
PYEOF
else
    echo "  WARN: $SERVER_ARGS not found — skipping argparse patch"
fi

echo "[3c/4] Installing sgl-kernel ABI compatibility stub (PyTorch 2.7 / sgl-kernel 0.3.x mismatch)..."
SGL_KERNEL_INIT="$SITE/sgl_kernel/__init__.py"
if [ -f "$SGL_KERNEL_INIT" ]; then
    cp "$REPO/directkv/sgl_kernel_stub.py" "$SGL_KERNEL_INIT"
    echo "  sgl_kernel/__init__.py replaced with ABI-safe stub"
else
    echo "  WARN: sgl_kernel not found at $SGL_KERNEL_INIT — skipping stub install"
fi

echo "[3d/4] Patching torchao_utils to guard against import crash on empty config..."
TORCHAO_UTILS="$SITE/sglang/srt/layers/torchao_utils.py"
if [ -f "$TORCHAO_UTILS" ]; then
    python3 - <<'PYEOF' "$TORCHAO_UTILS"
import pathlib, sys
p = pathlib.Path(sys.argv[1])
txt = p.read_text()
marker = '# Lazy import to suppress some warnings'
guard = '    if not torchao_config:\n        return model\n    '
if 'if not torchao_config:' not in txt:
    txt2 = txt.replace(marker, guard + marker, 1)
    if txt2 == txt:
        print(f"  WARN: could not find insertion point in {p}")
    else:
        p.write_text(txt2)
        print("  torchao_utils.py patched: early-return guard added")
else:
    print("  torchao_utils.py already patched")
PYEOF
else
    echo "  WARN: $TORCHAO_UTILS not found — skipping"
fi

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
