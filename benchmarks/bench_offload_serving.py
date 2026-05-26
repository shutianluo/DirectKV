#!/usr/bin/env python3
"""
Unified Benchmark Client — Experiment 1 (Figure 10)
=====================================================
Sends Poisson-distributed requests to a running SGLang (or compatible) server
and collects per-token latency (mean, P50, P90, P99) and throughput.

Supports four KV-cache backends: DirectKV, Pie, Neo, FlexGen.
Uses the same random seed, tokenizer, and warmup exclusion across all runs.

Usage
-----
# 1. Launch the server in a separate terminal:
python -m sglang.launch_server --model meta-llama/Llama-3.1-8B --tp 1 \\
       --attention-backend directkv --disable-radix-cache --disable-cuda-graph

# 2. Run the benchmark client:
python benchmarks/bench_offload_serving.py \\
    --system DirectKV \\
    --model-id meta-llama/Llama-3.1-8B \\
    --dataset sharegpt \\
    --request-rates 5 10 15 20 25 30 \\
    --num-requests 1000 \\
    --warmup-requests 100 \\
    --output-csv results/fig10_llama8b.csv

The script produces one CSV row per (system, model, request_rate) and appends
to --output-csv so you can run each system sequentially and concatenate.

CSV schema (matches comparison.md):
  system,model,request_rate,mean_latency,p50_latency,p90_latency,p99_latency,throughput_tps
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
WARMUP_REQUESTS = 100
NUM_REQUESTS = 1000

MODELS = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "opt-6.7b": "facebook/opt-6.7b",
    "opt-30b": "facebook/opt-30b",
}

SYSTEMS = [
    "DirectKV", "Pie", "Neo", "FlexGen", "NoOffload",
    "DirectKV_OPT6B", "Pie_OPT6B", "Neo_OPT6B",
    "DirectKV_OPT30B", "Pie_OPT30B", "FlexGen_OPT30B", "Neo_OPT30B",
    "FlexGen_LLaMA",
]

CSV_HEADER = [
    "system", "model", "dataset", "request_rate",
    "mean_latency", "p50_latency", "p90_latency", "p99_latency",
    "throughput_tps", "mean_ttft", "p99_ttft",
    "mean_e2e_latency", "p50_e2e_latency", "p90_e2e_latency", "p99_e2e_latency",
    "output_len",
    "num_completed", "num_failed",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestInput:
    prompt: str
    prompt_len: int
    output_len: int
    timestamp: Optional[float] = None


@dataclass
class RequestResult:
    success: bool = False
    latency: float = 0.0          # end-to-end seconds
    ttft: float = 0.0             # time to first token (s)
    itl: List[float] = field(default_factory=list)  # inter-token latencies (s)
    output_len: int = 0
    prompt_len: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_sharegpt(path: str, num_requests: int, tokenizer, context_len: int = 8192) -> List[RequestInput]:
    """Load ShareGPT conversations; filter by length."""
    from sglang.benchmark.utils import download_and_cache_hf_file, is_file_valid_json

    if not path or not is_file_valid_json(path):
        path = download_and_cache_hf_file(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
        )

    with open(path) as f:
        dataset = json.load(f)

    dataset = [
        d for d in dataset
        if len(d.get("conversations", d.get("conversation", []))) >= 2
    ]
    dataset = [
        (d.get("conversations", d.get("conversation", []))[0]["value"],
         d.get("conversations", d.get("conversation", []))[1]["value"])
        for d in dataset
    ]

    rng = random.Random(SEED)
    rng.shuffle(dataset)

    results: List[RequestInput] = []
    for prompt_text, completion_text in dataset:
        if len(results) >= num_requests:
            break
        p_ids = tokenizer.encode(prompt_text)
        c_ids = tokenizer.encode(completion_text)
        plen, olen = len(p_ids), len(c_ids)
        if plen < 4 or olen < 4:
            continue
        if plen + olen > context_len:
            continue
        results.append(RequestInput(prompt=prompt_text, prompt_len=plen, output_len=olen))

    print(f"[dataset] Loaded {len(results)} ShareGPT requests "
          f"(input: {sum(r.prompt_len for r in results)} tokens, "
          f"output: {sum(r.output_len for r in results)} tokens)")
    return results


def load_alpaca(num_requests: int, tokenizer, context_len: int = 8192) -> List[RequestInput]:
    """Load Alpaca instruction-following dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
    except Exception as e:
        print(f"[dataset] Failed to load Alpaca: {e}")
        print("[dataset] Falling back to synthetic prompts")
        return _synthetic_prompts(num_requests, tokenizer, 128, 64)

    items = list(ds)
    rng = random.Random(SEED)
    rng.shuffle(items)

    results: List[RequestInput] = []
    for item in items:
        if len(results) >= num_requests:
            break
        prompt = item.get("instruction", "") + "\n" + item.get("input", "")
        output = item.get("output", "")
        p_ids = tokenizer.encode(prompt.strip())
        c_ids = tokenizer.encode(output.strip())
        plen, olen = len(p_ids), len(c_ids)
        if plen < 4 or olen < 4:
            continue
        if plen + olen > context_len:
            continue
        results.append(RequestInput(prompt=prompt.strip(), prompt_len=plen, output_len=olen))

    print(f"[dataset] Loaded {len(results)} Alpaca requests")
    return results


def _synthetic_prompts(n: int, tokenizer, input_len: int, output_len: int) -> List[RequestInput]:
    """Fallback: synthetic prompts of fixed length."""
    base = "Summarize the following text in detail:\n" + "word " * (input_len - 10)
    p_ids = tokenizer.encode(base)
    return [RequestInput(prompt=base, prompt_len=len(p_ids), output_len=output_len)
            for _ in range(n)]


# ---------------------------------------------------------------------------
# HTTP client — sends streaming requests to SGLang /generate
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    req: RequestInput,
    neo_compat: bool = False,
    neo_model: str = "",
) -> RequestResult:
    """Send one streaming request and collect timing metrics.

    neo_compat=True switches to OpenAI /v1/completions format used by Neo's
    swiftllm server.  The SGLang /generate format is used otherwise.
    """
    result = RequestResult(prompt_len=req.prompt_len)

    if neo_compat:
        # OpenAI-compatible payload for Neo / swiftllm
        payload = {
            "model": neo_model or "default",
            "prompt": req.prompt,
            "max_tokens": req.output_len,
            "temperature": 0.0,
            "stream": True,
        }
    else:
        payload = {
            "text": req.prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": req.output_len,
                "ignore_eos": True,
            },
            "stream": True,
        }

    st = time.perf_counter()
    ttft_set = False
    last_output_len = 0
    most_recent_ts = st

    try:
        timeout = aiohttp.ClientTimeout(total=600)
        async with session.post(api_url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}"
                return result

            if neo_compat:
                # Neo/swiftllm streaming: one raw token_id per line (plain text)
                async for line_bytes in resp.content:
                    line = line_bytes.strip()
                    if not line:
                        continue
                    try:
                        int(line)  # validate it's a token id
                    except ValueError:
                        continue
                    now = time.perf_counter()
                    if not ttft_set:
                        result.ttft = now - st
                        ttft_set = True
                    else:
                        result.itl.append(now - most_recent_ts)
                    most_recent_ts = now
                    last_output_len += 1
            else:
                async for chunk_bytes in resp.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    text = chunk_bytes.decode("utf-8")
                    if text.startswith("data: "):
                        text = text[6:]
                    if text == "[DONE]":
                        break

                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        continue

                    if "text" in data and data["text"]:
                        now = time.perf_counter()
                        cur_output_len = data.get("meta_info", {}).get("completion_tokens", 0)
                        if not ttft_set:
                            result.ttft = now - st
                            ttft_set = True
                        else:
                            num_new = cur_output_len - last_output_len
                            if num_new > 0:
                                gap = now - most_recent_ts
                                per_tok = gap / num_new
                                result.itl.extend([per_tok] * num_new)
                        most_recent_ts = now
                        last_output_len = cur_output_len

        result.latency = time.perf_counter() - st
        result.output_len = last_output_len
        result.success = last_output_len > 0

    except Exception as e:
        result.latency = time.perf_counter() - st
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Poisson request scheduler
# ---------------------------------------------------------------------------

async def run_benchmark(
    api_url: str,
    requests: List[RequestInput],
    request_rate: float,
    warmup_requests: int = 100,
    neo_compat: bool = False,
    neo_model: str = "",
) -> Tuple[List[RequestResult], float]:
    """
    Send requests at Poisson-distributed intervals.
    Returns (results_after_warmup, duration_seconds).
    """
    connector = aiohttp.TCPConnector(limit=512)
    session = aiohttp.ClientSession(connector=connector, read_bufsize=10 * 1024 * 1024)

    rng = np.random.RandomState(SEED)
    tasks: List[asyncio.Task] = []

    # -- Warmup phase ---
    print(f"  Warmup: {warmup_requests} requests...")
    warmup_tasks = []
    for i in range(min(warmup_requests, len(requests))):
        warmup_tasks.append(asyncio.create_task(
            send_request(session, api_url, requests[i], neo_compat=neo_compat, neo_model=neo_model)
        ))
    warmup_results = await asyncio.gather(*warmup_tasks)
    warmup_ok = sum(1 for r in warmup_results if r.success)
    print(f"  Warmup done: {warmup_ok}/{len(warmup_results)} succeeded")

    # -- Main phase ---
    main_requests = requests[warmup_requests:]
    if not main_requests:
        await session.close()
        return [], 0.0

    start_time = time.perf_counter()

    for i, req in enumerate(main_requests):
        task = asyncio.create_task(
            send_request(session, api_url, req, neo_compat=neo_compat, neo_model=neo_model)
        )
        tasks.append(task)
        if request_rate < float("inf") and i < len(main_requests) - 1:
            interval = rng.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)

    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time

    await session.close()
    return list(results), duration


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    results: List[RequestResult],
    duration_s: float,
) -> Dict[str, Any]:
    """Compute latency and throughput metrics from results."""
    successful = [r for r in results if r.success]
    if not successful:
        return {
            "mean_latency": 0, "p50_latency": 0, "p90_latency": 0, "p99_latency": 0,
            "throughput_tps": 0, "mean_ttft": 0, "p99_ttft": 0,
            "num_completed": 0, "num_failed": len(results),
        }

    # Per-token latency: TPOT = (latency - ttft) / (output_tokens - 1)
    tpots = []
    for r in successful:
        if r.output_len > 1:
            tpot = (r.latency - r.ttft) / (r.output_len - 1)
            tpots.append(tpot)

    ttfts = [r.ttft for r in successful]
    e2e_latencies = [r.latency for r in successful]
    total_output_tokens = sum(r.output_len for r in successful)

    tpot_arr = np.array(tpots) if tpots else np.array([0.0])
    ttft_arr = np.array(ttfts) if ttfts else np.array([0.0])
    e2e_arr = np.array(e2e_latencies) if e2e_latencies else np.array([0.0])

    output_lens = [r.output_len for r in successful if r.output_len > 0]
    mean_output_len = float(np.mean(output_lens)) if output_lens else 0.0

    return {
        "mean_latency": float(np.mean(tpot_arr)),
        "p50_latency": float(np.percentile(tpot_arr, 50)),
        "p90_latency": float(np.percentile(tpot_arr, 90)),
        "p99_latency": float(np.percentile(tpot_arr, 99)),
        "throughput_tps": total_output_tokens / duration_s if duration_s > 0 else 0,
        "mean_ttft": float(np.mean(ttft_arr)),
        "p99_ttft": float(np.percentile(ttft_arr, 99)),
        "mean_e2e_latency": float(np.mean(e2e_arr)),
        "p50_e2e_latency": float(np.percentile(e2e_arr, 50)),
        "p90_e2e_latency": float(np.percentile(e2e_arr, 90)),
        "p99_e2e_latency": float(np.percentile(e2e_arr, 99)),
        "output_len": mean_output_len,
        "num_completed": len(successful),
        "num_failed": len(results) - len(successful),
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv_row(path: str, row: Dict[str, Any]):
    """Append a row to the CSV, writing the header if the file is new."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Offload benchmark client (Experiment 1)")
    p.add_argument("--system", required=True, choices=SYSTEMS,
                   help="Backend system name")
    p.add_argument("--model-id", default="meta-llama/Llama-3.1-8B",
                   help="HuggingFace model ID (for tokenizer)")
    p.add_argument("--model-label", default=None,
                   help="Short model name for CSV (default: derived from model-id)")
    p.add_argument("--base-url", default="http://127.0.0.1:30000",
                   help="Server base URL")
    p.add_argument("--endpoint", default="/generate",
                   help="Generate endpoint path")

    p.add_argument("--dataset", default="sharegpt", choices=["sharegpt", "alpaca", "synthetic"],
                   help="Dataset to use")
    p.add_argument("--dataset-path", default="",
                   help="Path to ShareGPT JSON (auto-downloads if empty)")
    p.add_argument("--context-len", type=int, default=8192,
                   help="Max total tokens per request")
    p.add_argument("--input-len", type=int, default=2048,
                   help="Synthetic dataset: input prompt length in tokens (default 2048)")
    p.add_argument("--output-len", type=int, default=256,
                   help="Synthetic dataset: output length in tokens (default 256)")

    p.add_argument("--request-rates", type=float, nargs="+",
                   default=[5, 10, 15, 20, 25, 30],
                   help="Request rates to sweep (req/s)")
    p.add_argument("--num-requests", type=int, default=NUM_REQUESTS,
                   help="Total requests per rate (warmup + main)")
    p.add_argument("--warmup-requests", type=int, default=WARMUP_REQUESTS,
                   help="Requests to discard as warmup")

    p.add_argument("--output-csv", default="results/fig10_latency_vs_rate.csv",
                   help="Output CSV path (appended)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--runs", type=int, default=3,
                   help="Repetitions per configuration (report mean ± std)")
    p.add_argument("--neo-compat", action="store_true",
                   help="Use OpenAI /v1/completions format (for Neo/swiftllm server)")
    p.add_argument("--neo-model", default="",
                   help="Model name to pass in the OpenAI 'model' field (Neo only)")
    return p.parse_args()


async def main_async():
    args = parse_args()

    global SEED
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Model label
    model_label = args.model_label
    if not model_label:
        model_label = args.model_id.split("/")[-1].lower()

    # Load dataset
    total_needed = args.num_requests
    if args.dataset == "sharegpt":
        requests = load_sharegpt(args.dataset_path, total_needed, tokenizer, args.context_len)
    elif args.dataset == "alpaca":
        requests = load_alpaca(total_needed, tokenizer, args.context_len)
    else:
        requests = _synthetic_prompts(total_needed, tokenizer, args.input_len, args.output_len)

    if len(requests) < args.warmup_requests + 10:
        print(f"[ERROR] Only {len(requests)} requests loaded, need ≥ {args.warmup_requests + 10}")
        sys.exit(1)

    api_url = args.base_url.rstrip("/") + args.endpoint
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Auto-enable Neo compat when system is Neo and endpoint is /v1/completions
    _neo_systems = {"Neo", "Neo_OPT6B", "Neo_OPT30B"}
    neo_compat = args.neo_compat or (args.system in _neo_systems and "/v1" in args.endpoint)
    neo_model = args.neo_model or (args.model_id if neo_compat else "")

    print(f"\n{'='*70}")
    print(f"Offload Benchmark: system={args.system}, model={model_label}")
    print(f"  dataset={args.dataset}, input_len={args.input_len}, output_len={args.output_len}")
    print(f"  requests={len(requests)}, warmup={args.warmup_requests}, runs={args.runs}")
    print(f"  rates={args.request_rates}")
    print(f"  server={api_url}  neo_compat={neo_compat}")
    print(f"{'='*70}\n")

    for rate in args.request_rates:
        run_metrics = []
        for run_i in range(args.runs):
            print(f"[{args.system}] rate={rate} req/s, run {run_i+1}/{args.runs}")
            results, duration = await run_benchmark(
                api_url, requests, rate, args.warmup_requests,
                neo_compat=neo_compat, neo_model=neo_model,
            )
            m = compute_metrics(results, duration)
            run_metrics.append(m)
            print(f"  → completed={m['num_completed']}, failed={m['num_failed']}, "
                  f"TPOT_mean={m['mean_latency']*1000:.1f}ms, "
                  f"throughput={m['throughput_tps']:.0f} tok/s")

        # Average across runs
        avg = {}
        for key in ["mean_latency", "p50_latency", "p90_latency", "p99_latency",
                     "throughput_tps", "mean_ttft", "p99_ttft",
                     "mean_e2e_latency", "p50_e2e_latency", "p90_e2e_latency", "p99_e2e_latency",
                     "output_len"]:
            vals = [m.get(key, 0.0) for m in run_metrics]
            avg[key] = float(np.mean(vals))

        avg["num_completed"] = int(np.mean([m["num_completed"] for m in run_metrics]))
        avg["num_failed"] = int(np.mean([m["num_failed"] for m in run_metrics]))

        row = {
            "system": args.system,
            "model": model_label,
            "dataset": args.dataset,
            "request_rate": rate,
            **avg,
        }
        write_csv_row(args.output_csv, row)
        print(f"  [saved] → {args.output_csv}")

    print(f"\nDone. Results appended to {args.output_csv}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
