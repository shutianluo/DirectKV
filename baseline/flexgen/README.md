# FlexGen Baseline

FlexGen is a CPU/disk offload inference engine for large language models. This directory
contains a modified version of [FMInference/FlexGen](https://github.com/FMInference/FlexGen)
extended with a streaming token callback and an HTTP serving wrapper compatible with
the artifact evaluation benchmark harness.

## System Requirements

- Python 3.10+
- PyTorch ≥ 1.12
- Hugging Face Transformers ≥ 4.24

## Installation

```bash
pip install -e baseline/flexgen/
```

## Usage

### HTTP server (for `ae_reviewer.sh` benchmarks)

```bash
python baseline/flexgen/serve_flexgen.py \
    --model facebook/opt-6.7b \
    --path <parent-dir-of-opt-6.7b-np> \
    --percent 0 100 100 0 100 0 \
    --port 30001 \
    --max-new-tokens 256
```

The `--percent` flag controls GPU/CPU/disk weight and KV-cache placement:
`w_gpu w_cpu w_disk kv_gpu kv_cpu kv_disk` (values are percentages summing to 100
for each group). The above setting puts all weights on CPU and all KV on CPU.

The server exposes a `/generate` endpoint (SSE streaming) and a `/health` endpoint.

### Supported models

FlexGen supports OPT models. Weights must be converted to the NumPy format first:

```bash
python -c "
from flexllmgen.flex_opt import get_filename, np_weight_dir
# See FlexGen docs or convert_opt_np_to_hf.py in the parent repo
"
```

Pre-converted weights for the artifact evaluation are expected at:
- `weights/opt/opt-6.7b-np/` — OPT-6.7B in NumPy format
- `weights/opt/opt-30b-np/` — OPT-30B in NumPy format
- `weights/opt/opt-6.7b-hf/` — OPT-6.7B HuggingFace format (tokenizer)

## Modifications vs upstream FlexGen

This directory extends `FMInference/FlexGen` (commit `004ffef`) with:

- **`serve_flexgen.py`** — new file: FastAPI HTTP server that wraps the
  `OptLM` engine, exposing `/generate` (SSE streaming) and `/health` endpoints
  compatible with `bench_offload_serving.py`

- **`flexllmgen/flex_opt.py`** — added `token_callback` parameter to
  `OptLM.generate()`, called once per generated token. Used by `serve_flexgen.py`
  to stream tokens over SSE without buffering the full sequence.
