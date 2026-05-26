#!/usr/bin/env python3
"""
Long-Context Benchmark — Experiment 2 (Figure 11)
====================================================
Measures per-token latency and GPU memory saving (%) as sequence length
increases, relative to a pure-GPU no-offload baseline.

Generates fixed-length synthetic requests at 1K, 2K, 4K, 8K tokens.
GPU memory is sampled via nvidia-smi at 100ms intervals during the run.

Usage
-----
# 1. Run the pure-GPU baseline first to get NoOffload peak memory:
python benchmarks/bench_offload_longctx.py \\
    --system NoOffload --model-id facebook/opt-30b \\
    --output-csv results/fig11_longctx.csv

# 2. Run each offload system:
python benchmarks/bench_offload_longctx.py \\
    --system DirectKV --model-id facebook/opt-30b \\
    --output-csv results/fig11_longctx.csv

# Repeat for Pie, Neo, FlexGen.

CSV schema (matches comparison.md):
  system,model,seq_length,mean_latency_s,p50_latency_s,p90_latency_s,p99_latency_s,
  gpu_memory_peak_mb,memory_saving_pct,throughput_tps
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
SEQ_LENGTHS = [1024, 2048, 4096, 8192]
REQUEST_RATE = 10.0       # fixed rate for Experiment 2
NUM_REQUESTS = 200        # per sequence length
WARMUP_REQUESTS = 20
OUTPUT_LEN = 128          # fixed output length

SYSTEMS = [
    "NoOffload", "DirectKV", "Pie", "Neo", "FlexGen",
    "DirectKV_OPT6B", "Pie_OPT6B", "Neo_OPT6B",
    "DirectKV_OPT30B", "FlexGen_OPT30B", "Neo_OPT30B",
]

CSV_HEADER = [
    "system", "model", "seq_length",
    "mean_latency_s", "p50_latency_s", "p90_latency_s", "p99_latency_s",
    "mean_e2e_latency_s", "p50_e2e_latency_s", "p90_e2e_latency_s", "p99_e2e_latency_s",
    "gpu_memory_peak_mb", "memory_saving_pct",
    "throughput_tps",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestInput:
    prompt: str
    prompt_len: int
    output_len: int


@dataclass
class RequestResult:
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    output_len: int = 0
    prompt_len: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Synthetic prompt generation (fixed context length)
# ---------------------------------------------------------------------------

def make_fixed_length_prompts(
    target_len: int,
    num_requests: int,
    tokenizer,
    output_len: int = OUTPUT_LEN,
) -> List[RequestInput]:
    """Generate prompts of approximately target_len input tokens."""
    # Build a base prompt and truncate/pad to target length
    base_text = "Please summarize the following document in detail. " * 100
    base_ids = tokenizer.encode(base_text)

    # Repeat to reach target length
    while len(base_ids) < target_len + 100:
        base_ids = base_ids + base_ids

    results = []
    for _ in range(num_requests):
        # Use exactly target_len tokens (must be 64-aligned for SMPv2 kernel path).
        # No variation: target_len values (1024, 2048, 4096, 8192) are all 64-aligned.
        ids = base_ids[:target_len]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        # Re-encode; align the result down to nearest 64 to satisfy kernel constraint.
        re_ids = tokenizer.encode(prompt)
        aligned_len = (len(re_ids) // 64) * 64
        aligned_len = max(64, aligned_len)
        re_ids = re_ids[:aligned_len]
        prompt = tokenizer.decode(re_ids, skip_special_tokens=True)

        results.append(RequestInput(
            prompt=prompt,
            prompt_len=aligned_len,
            output_len=output_len,
        ))

    mean_len = np.mean([r.prompt_len for r in results])
    print(f"[longctx] Generated {len(results)} prompts, "
          f"target={target_len}, actual_mean={mean_len:.0f} tokens")
    return results


# ---------------------------------------------------------------------------
# GPU memory sampler (nvidia-smi based)
# ---------------------------------------------------------------------------

class GPUMemorySampler:
    """Sample GPU memory via nvidia-smi in a background thread."""

    def __init__(self, gpu_id: int = 0, interval_s: float = 0.1):
        self.gpu_id = gpu_id
        self.interval_s = interval_s
        self.samples: List[int] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if not self.samples:
            return {"peak_mb": 0, "mean_mb": 0, "samples": 0}
        return {
            "peak_mb": max(self.samples),
            "mean_mb": sum(self.samples) / len(self.samples),
            "samples": len(self.samples),
        }

    def _sample_loop(self):
        while self._running:
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     f"--id={self.gpu_id}",
                     "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                val = result.stdout.strip().split("\n")[0].strip()
                if val.isdigit():
                    self.samples.append(int(val))
            except Exception:
                pass
            time.sleep(self.interval_s)


# ---------------------------------------------------------------------------
# HTTP client (reused from bench_offload_serving.py)
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    req: RequestInput,
    neo_compat: bool = False,
) -> RequestResult:
    result = RequestResult(prompt_len=req.prompt_len)

    if neo_compat:
        payload = {
            "prompt": req.prompt,
            "max_tokens": req.output_len,
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

    try:
        timeout = aiohttp.ClientTimeout(total=600)
        async with session.post(api_url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}"
                return result

            if neo_compat:
                # Neo streams raw token IDs, one integer per line
                async for line_bytes in resp.content:
                    line = line_bytes.strip()
                    if not line:
                        continue
                    try:
                        int(line)
                    except ValueError:
                        continue
                    now = time.perf_counter()
                    if not ttft_set:
                        result.ttft = now - st
                        ttft_set = True
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
                        if not ttft_set:
                            result.ttft = time.perf_counter() - st
                            ttft_set = True
                        cur_len = data.get("meta_info", {}).get("completion_tokens", 0)
                        last_output_len = cur_len

        result.latency = time.perf_counter() - st
        result.output_len = last_output_len
        result.success = last_output_len > 0
    except Exception as e:
        result.latency = time.perf_counter() - st
        result.error = str(e)
    return result


async def run_benchmark(
    api_url: str,
    requests: List[RequestInput],
    request_rate: float,
    warmup_requests: int,
    neo_compat: bool = False,
) -> Tuple[List[RequestResult], float]:
    connector = aiohttp.TCPConnector(limit=256)
    session = aiohttp.ClientSession(connector=connector, read_bufsize=10 * 1024 * 1024)
    rng = np.random.RandomState(SEED)

    # Warmup
    warmup_tasks = []
    for i in range(min(warmup_requests, len(requests))):
        warmup_tasks.append(asyncio.create_task(send_request(session, api_url, requests[i], neo_compat=neo_compat)))
    await asyncio.gather(*warmup_tasks)

    # Main
    main_reqs = requests[warmup_requests:]
    if not main_reqs:
        await session.close()
        return [], 0.0

    tasks = []
    start = time.perf_counter()
    for i, req in enumerate(main_reqs):
        tasks.append(asyncio.create_task(send_request(session, api_url, req, neo_compat=neo_compat)))
        if request_rate < float("inf") and i < len(main_reqs) - 1:
            interval = rng.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)

    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start
    await session.close()
    return list(results), duration


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[RequestResult], duration_s: float) -> Dict[str, Any]:
    successful = [r for r in results if r.success]
    if not successful:
        return {"mean_latency_s": 0, "p50_latency_s": 0, "p90_latency_s": 0,
                "p99_latency_s": 0, "throughput_tps": 0,
                "num_completed": 0, "num_failed": len(results)}

    # TPOT = (latency - ttft) / (output_tokens - 1)
    tpots = []
    for r in successful:
        if r.output_len > 1:
            tpots.append((r.latency - r.ttft) / (r.output_len - 1))

    e2e_latencies = [r.latency for r in successful]
    tpot_arr = np.array(tpots) if tpots else np.array([0.0])
    e2e_arr = np.array(e2e_latencies) if e2e_latencies else np.array([0.0])
    total_out = sum(r.output_len for r in successful)

    return {
        "mean_latency_s": float(np.mean(tpot_arr)),
        "p50_latency_s": float(np.percentile(tpot_arr, 50)),
        "p90_latency_s": float(np.percentile(tpot_arr, 90)),
        "p99_latency_s": float(np.percentile(tpot_arr, 99)),
        "mean_e2e_latency_s": float(np.mean(e2e_arr)),
        "p50_e2e_latency_s": float(np.percentile(e2e_arr, 50)),
        "p90_e2e_latency_s": float(np.percentile(e2e_arr, 90)),
        "p99_e2e_latency_s": float(np.percentile(e2e_arr, 99)),
        "throughput_tps": total_out / duration_s if duration_s > 0 else 0,
        "num_completed": len(successful),
        "num_failed": len(results) - len(successful),
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv_row(path: str, row: Dict[str, Any]):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})


def load_baseline_memory(csv_path: str, model: str) -> Dict[int, float]:
    """Load NoOffload peak GPU memory from CSV for computing memory_saving_pct."""
    baseline = {}
    if not os.path.exists(csv_path):
        return baseline
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("system") == "NoOffload" and row.get("model") == model:
                sl = int(row["seq_length"])
                mem = float(row["gpu_memory_peak_mb"])
                baseline[sl] = mem
    return baseline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Long-context benchmark (Experiment 2)")
    p.add_argument("--system", required=True, choices=SYSTEMS)
    p.add_argument("--model-id", default="facebook/opt-30b")
    p.add_argument("--model-label", default=None)
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--endpoint", default="/generate")
    p.add_argument("--gpu-id", type=int, default=0, help="GPU ID for nvidia-smi sampling")

    p.add_argument("--seq-lengths", type=int, nargs="+", default=SEQ_LENGTHS)
    p.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    p.add_argument("--request-rate", type=float, default=REQUEST_RATE)
    p.add_argument("--num-requests", type=int, default=NUM_REQUESTS)
    p.add_argument("--warmup-requests", type=int, default=WARMUP_REQUESTS)
    p.add_argument("--runs", type=int, default=3)

    p.add_argument("--output-csv", default="results/fig11_longctx.csv")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--neo-compat", action="store_true",
                   help="Use Neo/swiftllm streaming format (raw token IDs per line)")
    return p.parse_args()


async def main_async():
    args = parse_args()

    global SEED
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    model_label = args.model_label or args.model_id.split("/")[-1].lower()
    api_url = args.base_url.rstrip("/") + args.endpoint
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Load baseline memory if available (for computing memory_saving_pct)
    baseline_mem = load_baseline_memory(args.output_csv, model_label)

    print(f"\n{'='*70}")
    print(f"Long-Context Benchmark: system={args.system}, model={model_label}")
    print(f"  seq_lengths={args.seq_lengths}, output_len={args.output_len}")
    print(f"  rate={args.request_rate} req/s, requests={args.num_requests}, "
          f"warmup={args.warmup_requests}, runs={args.runs}")
    print(f"  baseline_memory={baseline_mem or 'not yet measured'}")
    print(f"{'='*70}\n")

    for seq_len in args.seq_lengths:
        total = args.num_requests + args.warmup_requests
        prompts = make_fixed_length_prompts(seq_len, total, tokenizer, args.output_len)

        run_metrics = []
        run_mems = []

        for run_i in range(args.runs):
            print(f"[{args.system}] seq_len={seq_len}, run {run_i+1}/{args.runs}")

            # Start GPU memory sampling
            mem_sampler = GPUMemorySampler(gpu_id=args.gpu_id, interval_s=0.1)
            mem_sampler.start()

            results, duration = await run_benchmark(
                api_url, prompts, args.request_rate, args.warmup_requests,
                neo_compat=args.neo_compat,
            )

            mem_info = mem_sampler.stop()
            m = compute_metrics(results, duration)
            run_metrics.append(m)
            run_mems.append(mem_info["peak_mb"])

            print(f"  → completed={m['num_completed']}, TPOT={m['mean_latency_s']*1000:.1f}ms, "
                  f"GPU_peak={mem_info['peak_mb']}MB, "
                  f"throughput={m['throughput_tps']:.0f} tok/s")

        # Average
        avg_metrics = {}
        for key in ["mean_latency_s", "p50_latency_s", "p90_latency_s", "p99_latency_s",
                     "mean_e2e_latency_s", "p50_e2e_latency_s", "p90_e2e_latency_s", "p99_e2e_latency_s",
                     "throughput_tps"]:
            avg_metrics[key] = float(np.mean([m[key] for m in run_metrics]))

        avg_peak_mb = float(np.mean(run_mems)) if run_mems else 0.0

        # Memory saving relative to NoOffload baseline
        mem_saving = 0.0
        if seq_len in baseline_mem and baseline_mem[seq_len] > 0:
            mem_saving = (baseline_mem[seq_len] - avg_peak_mb) / baseline_mem[seq_len] * 100.0
            mem_saving = max(0.0, mem_saving)
        elif args.system == "NoOffload":
            mem_saving = 0.0  # baseline itself

        row = {
            "system": args.system,
            "model": model_label,
            "seq_length": seq_len,
            **avg_metrics,
            "gpu_memory_peak_mb": avg_peak_mb,
            "memory_saving_pct": round(mem_saving, 1),
        }
        write_csv_row(args.output_csv, row)
        print(f"  [saved] → {args.output_csv}")

    print(f"\nDone. Results appended to {args.output_csv}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
