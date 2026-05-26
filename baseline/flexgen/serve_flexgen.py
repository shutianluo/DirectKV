#!/usr/bin/env python3
"""
FlexGen Online Serving HTTP Server
===================================
Wraps OptLM in a FastAPI server that speaks the same /generate SSE protocol
as SGLang, allowing bench_offload_serving.py to benchmark FlexGen alongside
DirectKV, Pie, and NoOffload.

Usage
-----
python FlexGen/serve_flexgen.py \
    --model facebook/opt-6.7b \
    --path weights/opt \
    --percent 0 100 100 0 100 0 \
    --port 30001

The --path argument should be the *parent* of the <model>-np directory.
For example, if weights live at weights/opt/opt-6.7b-np/, pass --path weights/opt.

Endpoint contract (matches SGLang /generate)
---------------------------------------------
POST /generate
  Request:  {"text": "...", "sampling_params": {"max_new_tokens": N,
              "temperature": 0.0, "ignore_eos": false}, "stream": true}
  Response: SSE stream
              data: {"text": "...", "meta_info": {"completion_tokens": N}}\n\n
              ...
              data: [DONE]\n\n

GET /health  →  {"status": "ok"} (HTTP 503 while loading)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Module-level state (populated in lifespan)
# ---------------------------------------------------------------------------

_model = None          # OptLM instance
_tokenizer = None      # HF AutoTokenizer
_gen_lock = threading.Lock()   # serialises all generate() calls
_args = None           # parsed CLI args (used by tests to check config)
_ready = False         # set True once model is fully loaded

log = logging.getLogger("serve_flexgen")


# ---------------------------------------------------------------------------
# Lifespan: load model at startup, clean up at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _ready
    log.info("Loading model %s from %s …", _args.model, _args.path)
    t0 = time.perf_counter()

    # Import FlexGen internals here so the import error surfaces at startup
    import os
    sys.path.insert(0, str(Path(__file__).parent))
    from flexllmgen.flex_opt import (ExecutionEnv, OptLM, Policy,
                                     CompressionConfig, get_opt_config)

    env = ExecutionEnv.create(_args.offload_dir)

    policy = Policy(
        gpu_batch_size=_args.gpu_batch_size,
        num_gpu_batches=1,
        w_gpu_percent=_args.percent[0],
        w_cpu_percent=_args.percent[1],
        cache_gpu_percent=_args.percent[2],
        cache_cpu_percent=_args.percent[3],
        act_gpu_percent=_args.percent[4],
        act_cpu_percent=_args.percent[5],
        overlap=True,
        sep_layer=True,
        pin_weight=True,
        cpu_cache_compute=False,
        attn_sparsity=1.0,
        compress_weight=False,
        comp_weight_config=CompressionConfig(
            num_bits=4, group_size=64, group_dim=0, symmetric=False),
        compress_cache=False,
        comp_cache_config=CompressionConfig(
            num_bits=4, group_size=64, group_dim=2, symmetric=False),
    )

    config = get_opt_config(_args.model)

    # Validate weight directory exists before attempting load (prevents silent HF download)
    weight_np_dir = os.path.join(
        os.path.abspath(os.path.expanduser(_args.path)),
        f"{config.name}-np"
    )
    if not os.path.isdir(weight_np_dir):
        log.error("Weight directory not found: %s", weight_np_dir)
        env.close_copy_threads()
        sys.exit(1)

    _model = OptLM(config, env, _args.path, policy)
    _tokenizer = AutoTokenizer.from_pretrained(_args.model, padding_side="left")
    _tokenizer.add_bos_token = False

    log.info("Model ready in %.1fs", time.perf_counter() - t0)
    _ready = True
    yield

    log.info("Shutting down …")
    if _model is not None:
        _model.env.close_copy_threads()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="FlexGen Serving", lifespan=lifespan)


@app.get("/health")
def health():
    if not _ready:
        return Response(content='{"status":"loading"}',
                        media_type="application/json", status_code=503)
    return {"status": "ok"}


@app.post("/generate")
async def generate_endpoint(request: Request):
    if not _ready:
        return Response(content='{"error":"model not ready"}',
                        media_type="application/json", status_code=503)

    body = await request.json()
    prompt: str = body.get("text", "")
    params: dict = body.get("sampling_params", {})
    max_new_tokens: int = min(
        int(params.get("max_new_tokens", _args.max_new_tokens)),
        _args.max_new_tokens,
    )
    ignore_eos: bool = bool(params.get("ignore_eos", False))
    stop_token: Optional[int] = (
        None if ignore_eos else _tokenizer.eos_token_id
    )

    # Tokenise prompt
    input_ids: list[int] = _tokenizer.encode(prompt)
    if not input_ids:
        return Response(content='{"error":"empty prompt"}',
                        media_type="application/json", status_code=400)

    # asyncio.Queue bridges the generation thread and this coroutine.
    # The generation thread deposits token IDs; None is the sentinel.
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[Optional[int]] = asyncio.Queue()

    def _token_callback(token_id: int) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, token_id)

    def _run_generate() -> None:
        try:
            with _gen_lock:
                _model.generate(
                    [input_ids],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    stop=stop_token,
                    token_callback=_token_callback,
                )
        except Exception as exc:
            log.exception("generate() failed: %s", exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Launch generation in a background thread so the event loop stays live.
    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()

    async def _stream() -> None:
        generated_ids: list[int] = []
        while True:
            token_id = await queue.get()
            if token_id is None:
                break
            generated_ids.append(token_id)
            text = _tokenizer.decode(generated_ids, skip_special_tokens=True)
            chunk = {
                "text": text,
                "meta_info": {"completion_tokens": len(generated_ids)},
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FlexGen online serving")
    p.add_argument("--model", default="facebook/opt-6.7b",
                   help="OPT model name (e.g. facebook/opt-6.7b)")
    p.add_argument("--path", default="weights/opt",
                   help="Parent dir of the <model>-np weight directory")
    p.add_argument("--percent", type=int, nargs=6,
                   default=[0, 100, 100, 0, 100, 0],
                   metavar=("W_GPU", "W_CPU", "KV_GPU", "KV_CPU",
                            "ACT_GPU", "ACT_CPU"),
                   help="Tensor placement percentages")
    p.add_argument("--gpu-batch-size", type=int, default=1)
    p.add_argument("--port", type=int, default=30001)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Hard cap on output length per request")
    p.add_argument("--offload-dir", default="~/flexllmgen_offload_dir",
                   help="Disk offload directory (created if absent)")
    return p.parse_args(argv)


def main(argv=None):
    global _args
    _args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("Starting FlexGen server on %s:%d", _args.host, _args.port)
    uvicorn.run(
        app,
        host=_args.host,
        port=_args.port,
        log_level="info",
        workers=1,
    )


if __name__ == "__main__":
    main()
