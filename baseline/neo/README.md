# Neo Baseline (SwiftLLM + PACPU)

Neo is a CPU-offload LLM serving system built on SwiftLLM. This directory contains
the modified SwiftLLM codebase extended for GH200 (aarch64) and OPT-model support,
plus the pre-compiled PACPU shared libraries used by the artifact evaluation.

## System Requirements

- NVIDIA GH200 (aarch64) — pre-built `.so` files target `aarch64-linux-gnu`
- Python 3.10
- CUDA 12.x
- `numactl` — `sudo apt-get install -y numactl`

## Installation

### 1. Install the SwiftLLM Python package

```bash
pip install -e baseline/neo/
```

### 2. Build and install the CUDA extension

```bash
pip install -e baseline/neo/csrc/
```

This compiles `swiftllm_c` (block-swapping, linear, and small CUDA kernels)
using `torch.utils.cpp_extension`. Requires CUDA 12.x and `pybind11`.

### 3. Locate the pre-compiled PACPU libraries

The PACPU kernel is a pre-compiled ARM/ISPC shared library that accelerates
CPU-side attention during KV offloading. Pre-built binaries for GH200 (aarch64)
are in `pacpu/prebuilt/`:

```
baseline/neo/pacpu/prebuilt/
  libpacpu-llama3_8b-tp1.so   # LLaMA-3-8B TP=1
  libpacpu-opt_6_7b-tp1.so    # OPT-6.7B  TP=1
  libpacpu-opt_30b-tp1.so     # OPT-30B   TP=1
```

Pass the appropriate library path to `--library-path` when launching the server.

### 4. (Optional) Rebuild PACPU from source

If you need to compile PACPU for a different target:

```bash
cd baseline/neo/pacpu/
bash build_arm.sh          # aarch64 (GH200)
# or
bash build.sh              # x86_64 with ISPC
```

## Usage

```bash
numactl -N 0 -m 0 python -m swiftllm.server.api_server \
    --port 8000 \
    --model-path <path-to-llama-3-8b> \
    --block-size 16 \
    --max-blocks-per-seq 1250 \
    --max-seqs-in-block-table 16 \
    --max-batch-size 16 \
    --max-tokens-in-batch 8192 \
    --tensor-parallel-degree 1 \
    --num-gpu-blocks-override 512 \
    --swap-space 20 \
    --library-path baseline/neo/pacpu/prebuilt/libpacpu-llama3_8b-tp1.so \
    --profile-result-path /tmp/neo_profile/ \
    --extra-layer-for-cprf
```

The server exposes an OpenAI-compatible `/v1/completions` endpoint on port 8000.

## Modifications vs upstream NEO

The SwiftLLM code in this directory extends
[`MachineLearningSystem/25MLSYS-NEO`](https://github.com/MachineLearningSystem/25MLSYS-NEO)
with the following changes:

- **ARM/GH200 PACPU support** — `pacpu/CMakeLists_arm.txt`, `pacpu/build_arm.sh`,
  `pacpu/pacpu_arm.h`, and corresponding changes in `pacpu/core.h`, `pacpu/dtype.h`,
  `pacpu/pacpu.cpp`
- **OPT model family** — `swiftllm/opt_model_config.py`,
  `swiftllm/worker/opt_model.py`, `swiftllm/worker/opt_weight.py`,
  `swiftllm/worker/layers/opt_transformer_layer.py`
- **Engine and executor fixes** — `swiftllm/server/engine.py`,
  `swiftllm/server/executor.py`, `swiftllm/worker/layers/transformer_layer.py`
- **Model config extensions** — `swiftllm/model_config.py`
