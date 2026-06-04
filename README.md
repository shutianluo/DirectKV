# DirectKV

DirectKV is a CPU-offload attention backend for [SGLang](https://github.com/sgl-project/sglang). It stores the KV cache in CPU-pinned memory instead of GPU HBM, enabling LLM serving with context lengths that far exceed GPU memory limits.

The core kernel (`directKV_kernel.cuh`) fuses K/V projection, Neox RoPE rotation, and multi-head attention into a single SM90a warp-group MMA kernel. KV tokens are loaded directly from CPU-pinned buffers via TMA, bypassing the GPU memory hierarchy.

## Paper

**No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs**  
Shutian Luo, Haiying Sheng  
*20th USENIX Symposium on Operating Systems Design and Implementation (OSDI '26)*

```bibtex
@inproceedings{directkv-osdi26,
  title     = {No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs},
  author    = {Shutian Luo and Haiying Sheng},
  booktitle = {Proceedings of the 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI '26)},
  year      = {2026},
  publisher = {USENIX Association},
}
```

## Artifact

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20385630.svg)](https://doi.org/10.5281/zenodo.20385630)

The archived artifact is available on Zenodo: <https://doi.org/10.5281/zenodo.20385630>

## License

This artifact is released under the [MIT License](LICENSE).

---

## Hardware

The paper's performance target is the **NVIDIA GH200 Grace-Hopper Superchip**, which connects CPU and GPU via NVLink-C2C (~900 GB/s). This bandwidth is what makes CPU-offloaded KV cache practical at long context lengths.

| | GH200 (NVLink-C2C) | H100 / H200 (PCIe) |
|---|---|---|
| Kernel correctness (unit tests) | ✓ | ✓ |
| SGLang integration / E2E functionality | ✓ | ✓ |
| Paper performance numbers | ✓ Full | PCIe bandwidth is ~18× lower |

> **You do not need a GH200 to evaluate correctness.** Any Hopper GPU (SM90) can run all unit tests and verify the SGLang integration.

---

## Artifact Evaluation (OSDI '26)

This section is the complete, self-contained guide for reproducing **Figure 10**
(online serving latency vs. request rate) and **Figure 11** (long-context decode
latency vs. sequence length).

> **Estimated runtime**: ~25 min (`--quick` mode) / ~2.5 hours (full run)

### Environment: prebuilt container (recommended) or build from scratch

This guide needs the DirectKV environment (SGLang + DirectKV, Pie, Neo, FlexGen, CUTLASS,
the CUDA toolchain). You can obtain it in two ways; **everything afterward — Steps 5–9 —
is identical either way.** Steps 1–4 below set the environment up by hand; the container
ships it pre-built.

**Option A — Prebuilt container (recommended).** Pull the image matching your hardware
architecture and open a shell inside it; Steps 1–4 are already done, so skip straight to
Step 5. The code in the image is exactly this repository — DirectKV and all baselines are
installed from `install.sh` and the `baseline/` tree, with no hidden or divergent code.

| Hardware | Arch | Image |
|----------|------|-------|
| **GH200** Grace-Hopper (NVLink-C2C) — *paper performance target* | `aarch64` | `shutianluo93/directkv:osdi26-ae-arm` |
| **H100 / H200** or other Hopper x86 hosts (PCIe) | `x86_64` | `shutianluo93/directkv:osdi26-ae-x86` |

```bash
IMG=shutianluo93/directkv:osdi26-ae-x86      # or :osdi26-ae-arm on a GH200
docker pull $IMG
docker run --rm -it --gpus all $IMG /bin/bash
export WORKSPACE=$(pwd)
# inside the container:  cd /workspace/DirectKV   →  then continue at Step 5 / Step 6
```

**Option B — Build from scratch.** Follow **Steps 0–4** below to create the `ae_python`
venv and install everything yourself, then continue with Step 5.

### What is reproduced

| Figure | Description | Systems | Script |
|--------|-------------|---------|--------|
| Fig 10(a) | TPOT vs. request rate — LLaMA-3-8B | DirectKV, Pie, Neo | `tests/ae_reviewer.sh` |
| Fig 10(b) | TPOT vs. request rate — OPT-6.7B | DirectKV, Pie, Neo, FlexGen | `tests/ae_reviewer.sh` |
| Fig 10(c) | TPOT vs. request rate — OPT-30B | DirectKV, FlexGen, Neo | `tests/ae_reviewer.sh` |
| Fig 11(a) | Decode latency vs. sequence length | DirectKV, Pie, FlexGen | `tests/ae_reviewer.sh` |
| Fig 11(b) | GPU memory saving % vs. seq length | DirectKV vs. baseline | `tests/ae_reviewer.sh` |

`tests/ae_reviewer.sh` runs all benchmarks end-to-end, writes raw CSV data to `results/<timestamp>/`, and generates `plots/figure10.pdf` and `plots/figure11.pdf` automatically. Step 9 below describes the output layout and the mapping from CSV rows to figure panels.

---

### Step 0 — Clone and set workspace

```bash
git clone https://github.com/shutianluo/DirectKV.git
cd DirectKV
export WORKSPACE=$(pwd)
```

---

### Step 1 — Create the `ae_python` virtual environment

All steps must run inside a dedicated venv so packages stay isolated in `$WORKSPACE/ae_python`.

```bash
python3 -m venv $WORKSPACE/ae_python
source $WORKSPACE/ae_python/bin/activate
which python3   # → $WORKSPACE/ae_python/bin/python3
```

Keep this venv active for every subsequent step.

---

### Step 2 — Install Python dependencies

```bash
# DirectKV / Pie backend
pip install sglang[all]==0.4.9.post6

# Neo / SwiftLLM (skip on x86; use --skip-neo in later steps)
pip install -e $WORKSPACE/baseline/neo/

# FlexGen
pip install -e $WORKSPACE/baseline/flexgen/

# Benchmark and plot utilities
pip install matplotlib pandas numpy fastapi uvicorn requests huggingface_hub
```

```bash
sudo apt-get install -y numactl
```


---

### Step 3 — CUTLASS headers

DirectKV's CUDA kernel is JIT-compiled on first use and requires SM90-capable CUTLASS headers.

```bash
git clone --branch v3.9.2 --depth 1 \
    https://github.com/NVIDIA/cutlass.git $WORKSPACE/cutlass
export CUTLASS_INCLUDE_DIR=$WORKSPACE/cutlass/include
```

---

### Step 4 — Install DirectKV into SGLang

```bash
bash $WORKSPACE/install.sh
```

`install.sh` does the following automatically:

| Action | Source → Destination |
|--------|----------------------|
| Copy DirectKV backends | `directkv/directkv_backend.py`, `directkv_smpv2_backend.py` → `sglang/srt/layers/attention/` |
| Copy JIT kernel wrappers | `directkv/kernel/*.cu` + `__init__.py` → `sglang/srt/layers/attention/directkv_kernel/` |
| Copy per-request KV pool | `directkv/directkv_request_pool.py` → `sglang/srt/layers/attention/` |
| Copy global KV pool | `directkv/directkv_pool.py` → `sglang/srt/mem_cache/` |
| Copy Pie offload module | `baseline/pie/kv_cache_offload.py` → `sglang/srt/layers/` |
| Copy OPT model | `directkv/opt_model.py` → `sglang/srt/models/opt.py` |
| Patch `server_args.py` | Adds `directkv` / `directkv-smpv2` to `--attention-backend` choices |
| Install `sgl_kernel` stub | Replaces broken `__init__.py` (PyTorch 2.7 / sgl-kernel 0.3.x ABI mismatch) |
| Patch `torchao_utils.py` | Adds early-return guard (prevents crash when `torchao_config=None`) |
| Create `neo_csrc.pth` | Adds `baseline/neo/csrc/` to `sys.path` so Neo's C extension is importable |

Verify:

```bash
python3 -c "from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool; print('OK')"
```

---

### Step 5 — Model weights

#### Option A — Use weights on the provided evaluation server

```bash
ln -sf /lambda/nfs/DirectKV/DirectKV_AE-main/weights $WORKSPACE/weights
```

Verify all five weight directories are present:

```bash
ls $WORKSPACE/weights/Llama/Llama-3-8B/
ls $WORKSPACE/weights/opt/opt-6.7b-hf/
ls $WORKSPACE/weights/opt/opt-6.7b-np/
ls $WORKSPACE/weights/opt/opt-30b-hf/
ls $WORKSPACE/weights/opt/opt-30b-np/
```

#### Option B — Download from Hugging Face

> **Disk space**: ~15 GB (LLaMA-3-8B) + 14 GB (OPT-6.7B HF) + 14 GB (OPT-6.7B NP)
> + 60 GB (OPT-30B HF) + 60 GB (OPT-30B NP) ≈ **163 GB total**

```bash
mkdir -p $WORKSPACE/weights/Llama $WORKSPACE/weights/opt
```

**LLaMA-3-8B** (requires HF account + Meta license):

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Meta-Llama-3-8B',
                  local_dir='$WORKSPACE/weights/Llama/Llama-3-8B')
"
```

**OPT-6.7B — HF format** (for SGLang / DirectKV / Pie):

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/opt-6.7b',
                  local_dir='$WORKSPACE/weights/opt/opt-6.7b-hf')
"
```

**OPT-6.7B — numpy format** (for FlexGen):

```bash
python3 -c "
from flexllmgen.opt_config import download_opt_weights
download_opt_weights('opt-6.7b', '$WORKSPACE/weights/opt/')
"
```

**OPT-30B — HF format** (for DirectKV + Neo OPT-30B panels):

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/opt-30b',
                  local_dir='$WORKSPACE/weights/opt/opt-30b-hf')
"
```

**OPT-30B — numpy format** (for FlexGen):

```bash
python3 -c "
from flexllmgen.opt_config import download_opt_weights
download_opt_weights('opt-30b', '$WORKSPACE/weights/opt/')
"
```

> OPT-30B is optional. Pass `--skip-opt30b` to skip Fig 10(c) and the OPT-30B panel of Fig 11.

---

### Step 6 — Correctness tests (~3 min, no server required)

Verifies that the DirectKV CUDA kernel and Pie offload module produce numerically correct outputs.

```bash
bash $WORKSPACE/tests/build_and_test.sh
```

Expected output (14 Python + 12 CUDA tests, all passing):

```
=== DirectKV Correctness Tests ===

[1/2] Python correctness test (test_directkv_correctness.py)...
..............
Ran 14 tests in ~7s
OK

[2/2] CUDA kernel test (test_directkv_cpu_kv.cu)...
SM-Parallel v2 CPU KV — Correctness Tests

── Prefill (all new KV) ──
pf-tiny          O_err=0.00006(OK)  K_err=0.00046(OK)  V_err=0.00043(OK)  → PASS
...
12 / 12 passed

=== All tests PASSED ===
```

`build_and_test.sh` automatically detects the GPU. On PCIe hardware it prints:

```
NOTE: Kernel is functionally correct on this GPU.
      CPU-KV bandwidth is PCIe-limited (~50 GB/s) vs NVLink-C2C (~900 GB/s).
      Correctness tests pass; end-to-end decode latency will differ from paper.
```

---

### Step 7 — Smoke test (~15 min, all four systems)

Verifies that all four servers start, accept requests, and return valid text before the full benchmark.

```bash
bash $WORKSPACE/tests/smoke_tests.sh
```

Optional flags:
- `--skip-neo` — skip Neo (required on x86; aarch64 pacpu `.so` only)
- `--skip-flexgen` — skip FlexGen (if OPT weights not yet available)

Expected final output:

```
  ┌──────────────────────────────┬──────────────┐
  │ System                       │ Status       │
  ├──────────────────────────────┼──────────────┤
  │ DirectKV                     │ ✓ PASS       │
  │ Pie                          │ ✓ PASS       │
  │ Neo                          │ ✓ PASS       │
  │ FlexGen                      │ ✓ PASS       │
  └──────────────────────────────┴──────────────┘
  PASSED: 4 / 4
  ✓  All systems passed. Safe to run the full benchmark.
```

Do not proceed to Step 8 (benchmark) until all expected systems pass.

---

### Step 8 — Run the benchmark

#### Quick mode (~25 min, sanity check)

```bash
bash $WORKSPACE/tests/ae_reviewer.sh --quick
```

#### Full run (~2.5 h, reproduces paper figures)

```bash
bash $WORKSPACE/tests/ae_reviewer.sh
```

Available flags:

| Flag | Effect |
|------|--------|
| `--quick` | Reduced request rates and sequence lengths |
| `--skip-neo` | Skip all Neo sections |
| `--skip-flexgen` | Skip all FlexGen sections |
| `--skip-opt30b` | Skip OPT-30B panels (Figs 10c, 11 OPT-30B) |
| `--fig10-only` | Skip Figure 11 (long-context) |

---

### Step 9 — Inspect results

Results are written to `results/<timestamp>/`:

```
results/2026-05-25_12-00/
  fig10_serving.csv     # TPOT / TTFT vs. request rate (all systems, all models)
  fig11_longctx.csv     # Decode latency + memory saving vs. sequence length
  plots/
    figure10.pdf        # 3-panel Figure 10 (LLaMA-3-8B, OPT-6.7B, OPT-30B)
    figure11.pdf        # 2-panel Figure 11 (latency + memory saving)
  logs/                 # Per-server and per-benchmark logs
```

#### Figure mapping

| Figure panel | CSV `system` label | Key claim |
|---|---|---|
| Fig 10(a) LLaMA-3-8B | `DirectKV`, `Pie`, `Neo` | DirectKV TPOT lowest under high load |
| Fig 10(b) OPT-6.7B | `DirectKV_OPT6B`, `Pie_OPT6B`, `Neo_OPT6B`, `FlexGen` | DirectKV sustains higher request rates |
| Fig 10(c) OPT-30B | `DirectKV_OPT30B`, `FlexGen_OPT30B`, `Neo_OPT30B` | DirectKV scales to 30B without OOM |
| Fig 11(a) latency | `DirectKV`, `Pie`, `DirectKV_OPT6B`, `FlexGen` | DirectKV TPOT grows sub-linearly |
| Fig 11(b) memory | all `DirectKV*` rows | ≥90% GPU memory freed for long contexts |

---

### Troubleshooting

**Neo startup takes ~2 minutes** — this is normal. Neo profiles the CPU kernel on first launch. Wait for `[wait] Neo ready`.

**Stale server from a previous run**

```bash
pkill -9 -f "sglang.launch_server"
pkill -9 -f "swiftllm.server.api_server"
pkill -9 -f "serve_flexgen"
```

**DirectKV CUDA graph capture errors at startup** — run with `--keep-cuda-graph-off` to isolate:

```bash
bash tests/smoke_tests.sh --keep-cuda-graph-off
```

The full benchmark already disables CUDA graphs, so this does not affect AE results.

---


## Repository Structure

```
DirectKV/
  csrc/                              # CUDA kernel headers
    directKV_kernel.cuh              # Main fused attention kernel (SM90a)
    proj_fused_kernel_traits_sm90.h  # CUTLASS MMA/TMA kernel traits
    softmax.h                        # Online softmax (from FlashAttention-3)
    utils.h                          # Shared device utilities

  directkv/                          # Python package — installed via install.sh
    directkv_backend.py              # SGLang AttentionBackend (Python path)
    directkv_smpv2_backend.py        # SM90 warp-group MMA backend
    directkv_pool.py                 # DirectKVTokenToKVPool (CPU-pinned KV pool)
    directkv_request_pool.py         # Per-request contiguous KV buffer
    directkv_startup_hook.py         # Auto-registration hook (loaded via .pth)
    directkv_startup.pth             # Triggers startup hook at Python init
    opt_model.py                     # OPT model registration for SGLang
    sgl_kernel_stub.py               # PyTorch-2.7-compatible sgl_kernel replacement
    kernel/
      __init__.py                    # JIT loader for CUDA extensions
      directkv_extension.cu          # PyBind11 wrapper — DirectKV kernel
      qcentric_extension.cu          # PyBind11 wrapper — Q-centric kernel
      smpv2_extension.cu             # PyBind11 wrapper — SM90 smpv2 kernel

  kernel/                            # SM90 kernel headers (used by JIT compiler)

  tests/
    build_and_test.sh                # Step 6 — kernel correctness tests
    smoke_tests.sh                   # Step 7 — smoke test for all four systems
    ae_reviewer.sh                   # Step 8 — full AE benchmark (~2.5 h)
    test_directkv_correctness.py     # Python correctness test (14 cases)
    test_directkv_cpu_kv.cu          # Standalone CUDA kernel test (12 cases)
    test_pie_offload.py              # Pie offload correctness test

  benchmarks/                        # Benchmark client scripts
  evaluation/                        # Plot-generation scripts

  baseline/
    neo/                             # Neo / SwiftLLM + aarch64 pre-built pacpu .so
    flexgen/                         # FlexGen (serve_flexgen.py + flexllmgen package)
    pie/
      kv_cache_offload.py            # Pie KV-offload module (PieKVPool, SwapEngine)
      README.md                      # Pie design notes and launch flags

  install.sh                         # Step 4 — installation script
  requirements.txt                   # Python dependency list
  README.md                          # This file (includes full AE guide)
  LICENSE
```

---

## Attribution

DirectKV builds on [FlashAttention-3](https://github.com/Dao-AILab/flash-attention) (Shah et al., 2024) and extends it from GPU-resident execution to a heterogeneous CPU–GPU setting on the GH200. FlashAttention-3 assumes the entire KV cache resides in GPU HBM; DirectKV removes that assumption by streaming KV tokens from CPU-pinned memory into the SM90a attention kernel via TMA, without staging through a GPU buffer.

The following components are taken or derived from FlashAttention-3:

- `csrc/softmax.h` and `csrc/utils.h` — taken directly from FlashAttention-3.
- `csrc/proj_fused_kernel_traits_sm90.h` — derived from FlashAttention-3's SM90 kernel traits, extended for fused K/V projection and TMA-based CPU→GPU movement.
- `directkv/kernel/qcentric_extension.cu` — wraps the Q-centric attention kernel from FlashAttention-3's research branch.

The DirectKV fused kernel (`csrc/directKV_kernel.cuh`) is original work.
