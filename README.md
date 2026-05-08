# DirectKV

DirectKV is a CPU-offload attention backend for [SGLang](https://github.com/sgl-project/sglang) that stores the KV cache in CPU-pinned memory instead of GPU HBM. This enables LLM serving with context lengths far exceeding GPU memory limits on GH200/SM90 hardware, using NVLink-C2C bandwidth to stream KV tokens from host memory into the SM90 kernel via TMA without copying through the CPU.

The core kernel (`directKV_kernel.cuh`) fuses K/V projection, Neox RoPE rotation, and multi-head attention into a single SM90a warp-group MMA kernel. KV tokens are loaded directly from CPU-pinned buffers via TMA, bypassing the GPU memory hierarchy.

## System Requirements

- NVIDIA GH200 (SM90a / `sm_90a` architecture) — required for warp-group MMA and TMA from CPU memory
- CUDA 12.x with `nvcc` supporting `-arch=sm_90a`
- Python 3.10+
- SGLang 0.4.9 (`sglang[all]==0.4.9.post6`)
- CUTLASS headers (SM90-capable; see Installation below)

## Installation

### 1. Install SGLang and Python dependencies

```bash
pip install sglang[all]==0.4.9.post6
pip install pybind11 numpy transformers
```

### 2. Get CUTLASS headers

DirectKV's CUDA kernel requires CUTLASS SM90 headers to compile.

**Option A (recommended):** Install via `deep-gemm`, which bundles CUTLASS:

```bash
pip install deep-gemm
```

**Option B:** Clone CUTLASS manually and set the include path:

```bash
git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass
export CUTLASS_INCLUDE_DIR=/opt/cutlass/include
```

### 3. Run the installer

```bash
bash install.sh
```

The installer copies the startup hook, backend module, and pool module into the appropriate SGLang site-packages locations. The CUDA kernel is JIT-compiled on first use via `torch.utils.cpp_extension.load`.

## Usage

Launch an SGLang server with the DirectKV backend:

```bash
python -m sglang.launch_server \
    --model-path <your-model> \
    --attention-backend directkv \
    --disable-radix-cache \
    --disable-cuda-graph
```

Key flags:
- `--attention-backend directkv` — selects the DirectKV backend
- `--disable-radix-cache` — required (DirectKV manages its own CPU-pinned KV pool)
- `--disable-cuda-graph` — recommended for prefill; decode CUDA graphs are supported
- `--chunked-prefill-size 64` — chunked prefill must be a multiple of 64

### Constraints

- `head_dim == 128` and `bfloat16` model weights are required
- Supported model families: OPT, LLaMA-3 (with GQA), Mistral
- MLA models (e.g. DeepSeek) are not supported

## Running Correctness Tests

```bash
cd tests/
bash build_and_test.sh
```

This runs two tests:
1. A Python test that checks KV pool round-trip correctness and attention output against `torch.nn.functional.scaled_dot_product_attention`
2. A standalone CUDA test that compiles and runs `test_directkv_cpu_kv.cu` to validate the SM90a kernel output against a reference

## Repository Structure

```
Github_Repo/
  csrc/                        # CUDA kernel headers (included by the .cu extension)
    directKV_kernel.cuh   # Main fused attention kernel (SM90a)
    proj_fused_kernel_traits_sm90.h       # CUTLASS MMA/TMA kernel traits
    softmax.h                             # Online softmax utilities
    utils.h                               # Shared device utilities

  directkv/                    # Python package — install via install.sh
    __init__.py
    directkv_backend.py        # SGLang AttentionBackend implementation
    directkv_pool.py           # DirectKVTokenToKVPool (CPU-pinned KV storage)
    directkv_request_pool.py   # Per-request contiguous KV pool for the kernel path
    directkv_startup_hook.py   # Auto-registration hook (injected at Python startup)
    directkv_startup.pth       # .pth file that triggers the hook on import
    kernel/
      __init__.py              # JIT loader for CUDA extensions
      directkv_extension.cu   # PyBind11 wrapper for the DirectKV kernel
      qcentric_extension.cu   # PyBind11 wrapper for the original Q-centric kernel

  tests/
    test_directkv_correctness.py   # Python correctness test
    test_directkv_cpu_kv.cu           # Standalone CUDA kernel correctness test
    build_and_test.sh              # Build and run all tests

  install.sh                   # Installation script
  requirements.txt             # Python dependencies
  README.md                    # This file
```
