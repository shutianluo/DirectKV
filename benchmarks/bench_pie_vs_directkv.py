#!/usr/bin/env python3
"""
Pie vs DirectKV vs GPU-only Benchmark
=======================================
Compares three KV cache strategies for multi-layer decode attention:

  1. **GPU-only**    — standard MHATokenToKVPool (all KV on GPU).
  2. **DirectKV**    — CPU-pinned KV pool, each decode gathers K/V to GPU.
  3. **Pie**         — GPU pool with async layer-level swap (Pie paper).

Metrics:
  - Per-step decode latency (ms) across context lengths.
  - GPU memory usage.
  - Stall count for Pie.
  - Throughput (tokens/s).

Usage:
    python benchmarks/bench_pie_vs_directkv.py                  # defaults
    python benchmarks/bench_pie_vs_directkv.py --num-layers 32  # custom
    python benchmarks/bench_pie_vs_directkv.py --csv out.csv    # save

Requirements: torch, CUDA GPU.
"""

import argparse
import math
import time
from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pie vs DirectKV benchmark")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--num-layers", type=int, default=16,
                   help="Number of transformer layers to simulate")
    p.add_argument("--num-q-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[128, 512, 1024, 2048],
                   help="Context lengths to sweep")
    p.add_argument("--expansion-ratio", type=float, default=1.3,
                   help="Pie expansion ratio (default 1.3×)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--csv", default=None)
    return p.parse_args()


# -----------------------------------------------------------------------
# Attention helper
# -----------------------------------------------------------------------

def decode_attention_one(q, k, v, scale, nqh, nkvh):
    """
    Single-token decode attention.
    q: (1, nqh, D)  GPU
    k: (S, nkvh, D) GPU
    v: (S, nkvh, D) GPU
    Returns: (1, nqh, D) GPU
    """
    groups = nqh // nkvh
    q4 = q.unsqueeze(0).permute(0, 2, 1, 3)   # (1, nqh, 1, D)
    k4 = k.permute(1, 0, 2).unsqueeze(0)       # (1, nkvh, S, D)
    v4 = v.permute(1, 0, 2).unsqueeze(0)
    if groups > 1:
        k4 = k4.repeat_interleave(groups, dim=1)
        v4 = v4.repeat_interleave(groups, dim=1)
    o = F.scaled_dot_product_attention(q4, k4, v4, scale=scale)
    return o.squeeze(0).permute(1, 0, 2)  # (1, nqh, D)


# -----------------------------------------------------------------------
# Benchmark: GPU-only baseline
# -----------------------------------------------------------------------

def bench_gpu_only(args, seq_len):
    dtype = getattr(torch, args.dtype)
    device = args.device
    nqh, nkvh, D = args.num_q_heads, args.num_kv_heads, args.head_dim
    bs, nl = args.batch_size, args.num_layers
    scale = 1.0 / math.sqrt(D)

    # Pre-allocate per-layer GPU KV
    shape = (seq_len, nkvh, D)
    kv_layers_k = [torch.randn(shape, device=device, dtype=dtype) for _ in range(nl)]
    kv_layers_v = [torch.randn(shape, device=device, dtype=dtype) for _ in range(nl)]
    q = torch.randn(bs, nqh, D, device=device, dtype=dtype)

    def run():
        for li in range(nl):
            for b in range(bs):
                decode_attention_one(q[b:b+1], kv_layers_k[li], kv_layers_v[li],
                                     scale, nqh, nkvh)

    # Warmup
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iters):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return min(times)


# -----------------------------------------------------------------------
# Benchmark: DirectKV
# -----------------------------------------------------------------------

def bench_directkv(args, seq_len):
    dtype = getattr(torch, args.dtype)
    device = args.device
    nqh, nkvh, D = args.num_q_heads, args.num_kv_heads, args.head_dim
    bs, nl = args.batch_size, args.num_layers
    scale = 1.0 / math.sqrt(D)

    shape = (seq_len, nkvh, D)
    # CPU-pinned KV
    cpu_k = [torch.randn(shape, dtype=dtype).pin_memory() for _ in range(nl)]
    cpu_v = [torch.randn(shape, dtype=dtype).pin_memory() for _ in range(nl)]
    q = torch.randn(bs, nqh, D, device=device, dtype=dtype)

    def run():
        for li in range(nl):
            k_gpu = cpu_k[li].to(device, non_blocking=True)
            v_gpu = cpu_v[li].to(device, non_blocking=True)
            for b in range(bs):
                decode_attention_one(q[b:b+1], k_gpu, v_gpu, scale, nqh, nkvh)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iters):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return min(times)


# -----------------------------------------------------------------------
# Benchmark: Pie (async layer-level swap)
# -----------------------------------------------------------------------

def bench_pie(args, seq_len):
    from sglang.srt.layers.kv_cache_offload import (
        MappingTable, SwapEngine, FIFOSwapController,
    )

    dtype = getattr(torch, args.dtype)
    device_str = args.device
    device = torch.device(device_str)
    nqh, nkvh, D = args.num_q_heads, args.num_kv_heads, args.head_dim
    bs, nl = args.batch_size, args.num_layers
    scale = 1.0 / math.sqrt(D)
    expansion = args.expansion_ratio

    m = max(0, round(nl * (1.0 - 1.0 / expansion)))
    m = min(m, nl - 2)
    num_gpu = nl - m

    shape = (seq_len, nkvh, D)
    gpu_k = [torch.randn(shape, device=device, dtype=dtype) for _ in range(nl)]
    gpu_v = [torch.randn(shape, device=device, dtype=dtype) for _ in range(nl)]
    cpu_k = [torch.zeros(shape, dtype=dtype, pin_memory=True) for _ in range(nl)]
    cpu_v = [torch.zeros(shape, dtype=dtype, pin_memory=True) for _ in range(nl)]

    mapping = MappingTable(nl, num_gpu)
    engine = SwapEngine(device)

    # Copy offloaded layers to CPU
    for lid in mapping.cpu_layers():
        cpu_k[lid].copy_(gpu_k[lid])
        cpu_v[lid].copy_(gpu_v[lid])

    ctrl = FIFOSwapController(
        mapping, engine, gpu_k, gpu_v, cpu_k, cpu_v
    )

    q = torch.randn(bs, nqh, D, device=device, dtype=dtype)

    def run():
        for li in range(nl):
            ctrl.on_layer_compute_start(li)
            for b in range(bs):
                decode_attention_one(q[b:b+1], gpu_k[li], gpu_v[li], scale, nqh, nkvh)

    # Warmup
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    ctrl.stall_count = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iters):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    stalls = ctrl.stall_count
    return min(times), stalls, m


# -----------------------------------------------------------------------
# Memory measurement
# -----------------------------------------------------------------------

def measure_gpu_memory(args, seq_len):
    dtype = getattr(torch, args.dtype)
    device = args.device
    nkvh, D, nl = args.num_kv_heads, args.head_dim, args.num_layers
    shape = (seq_len, nkvh, D)
    elem_bytes = 2  # fp16/bf16

    per_layer_bytes = seq_len * nkvh * D * elem_bytes * 2  # K + V
    gpu_only_bytes = nl * per_layer_bytes

    expansion = args.expansion_ratio
    m = max(0, round(nl * (1.0 - 1.0 / expansion)))
    m = min(m, nl - 2)
    pie_gpu_bytes = (nl - m) * per_layer_bytes  # only keep n-m on GPU

    return {
        "gpu_only_MB": gpu_only_bytes / 1e6,
        "directkv_gpu_MB": 0.0,  # all on CPU
        "pie_gpu_MB": pie_gpu_bytes / 1e6,
        "pie_cpu_MB": m * per_layer_bytes / 1e6,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"Pie vs DirectKV vs GPU-only Benchmark")
    print(f"  device={args.device}, dtype={args.dtype}")
    print(f"  layers={args.num_layers}, nqh={args.num_q_heads}, "
          f"nkvh={args.num_kv_heads}, D={args.head_dim}")
    print(f"  batch_size={args.batch_size}, expansion={args.expansion_ratio:.2f}×")
    print()

    # Header
    print(f"{'seq_len':>10}  {'gpu_only_ms':>12}  {'directkv_ms':>12}  "
          f"{'pie_ms':>10}  {'pie/gpu':>8}  {'dkv/gpu':>8}  "
          f"{'pie_m':>5}  {'stalls':>6}")
    print("-" * 95)

    results = []
    for sl in args.seq_lens:
        t_gpu = bench_gpu_only(args, sl)
        t_dkv = bench_directkv(args, sl)
        t_pie, stalls, m = bench_pie(args, sl)

        pie_ratio = t_pie / t_gpu if t_gpu > 0 else float("nan")
        dkv_ratio = t_dkv / t_gpu if t_gpu > 0 else float("nan")

        print(f"{sl:>10}  {t_gpu:>12.3f}  {t_dkv:>12.3f}  "
              f"{t_pie:>10.3f}  {pie_ratio:>8.2f}×  {dkv_ratio:>8.2f}×  "
              f"{m:>5}  {stalls:>6}")

        mem = measure_gpu_memory(args, sl)
        results.append({
            "seq_len": sl,
            "gpu_only_ms": t_gpu,
            "directkv_ms": t_dkv,
            "pie_ms": t_pie,
            "pie_over_gpu": pie_ratio,
            "directkv_over_gpu": dkv_ratio,
            "pie_m": m,
            "pie_stalls": stalls,
            **mem,
        })

    # Memory summary
    print()
    mem = measure_gpu_memory(args, args.seq_lens[-1])
    print(f"GPU memory (seq_len={args.seq_lens[-1]}):")
    print(f"  GPU-only  : {mem['gpu_only_MB']:.1f} MB")
    print(f"  DirectKV  : {mem['directkv_gpu_MB']:.1f} MB GPU (all on CPU)")
    print(f"  Pie       : {mem['pie_gpu_MB']:.1f} MB GPU + {mem['pie_cpu_MB']:.1f} MB CPU")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()
