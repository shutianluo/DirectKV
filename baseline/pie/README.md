# Pie Baseline

Pie is SGLang's built-in CPU KV-offload path using the **FlashInfer** attention
backend. It requires no additional installation beyond the SGLang dependency
already installed for DirectKV.

## Files

| File | Description |
|------|-------------|
| `kv_cache_offload.py` | Pie KV-offload module: `PieKVPool`, `SwapEngine`, `AdaptiveExpansionController`. Installed into SGLang by `install.sh` (`sglang/srt/layers/kv_cache_offload.py`). |

`install.sh` automatically copies `kv_cache_offload.py` into the active SGLang
installation. No manual steps are needed.

## How Pie offloads KV

Pie relies on SGLang's radix-cache eviction path: when the GPU KV pool is full,
SGLang serializes the oldest KV blocks to CPU memory. This is transparent to the
attention kernel (FlashInfer), which always operates on GPU tensors — host memory
is only used as an overflow buffer.

DirectKV, by contrast, streams KV tokens directly from CPU-pinned memory into
the SM90a attention kernel via TMA, without requiring GPU-resident copies.

## Server launch flags

```bash
python -m sglang.launch_server \
    --model-path <model-path> \
    --tp 1 \
    --port 30000 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.50 \
    --disable-radix-cache \
    --disable-cuda-graph \
    --max-running-requests 16 \
    --skip-server-warmup
```

## Key flags vs DirectKV

| Flag | DirectKV | Pie |
|------|----------|-----|
| `--attention-backend` | `directkv-smpv2` | `flashinfer` |
| `--disable-cuda-graph` | not set (CUDA graphs enabled) | required |
| `--skip-server-warmup` | not set | recommended (avoids 4096-request warmup) |

## SGLang version

Both Pie and DirectKV require `sglang[all]==0.4.9.post6`. See the top-level
`install.sh` for the full installation procedure.
