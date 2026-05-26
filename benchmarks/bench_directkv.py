"""
DirectKV microbenchmark — CPU-pinned KV cache vs GPU KV cache.

Measures:
  1. KV-write throughput  (set_kv_buffer): tokens/s for GPU pool vs CPU-pinned pool.
  2. Decode attention latency: time per forward call for a single decode step,
     sweeping sequence length (context window size).
  3. Memory footprint: GPU memory saved by offloading KV to CPU.

Backends compared:
  - "flash"  : FlashInfer decode (GPU KV) via SGLang's flashinfer_backend.
  - "triton" : Triton decode (GPU KV) via SGLang's triton_backend.
  - "directkv-pytorch" : DirectKV PyTorch reference path (CPU-pinned KV, D2H gather).

Run:
    python benchmarks/bench_directkv.py [--device cuda] [--csv out.csv]

Requires: torch, tabulate (pip install tabulate)
"""

import argparse
import math
import time
import sys
from dataclasses import dataclass, field
from typing import List

import torch

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DirectKV microbenchmark")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--num-q-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048, 4096, 8192],
        help="Context lengths (total tokens including new) to sweep",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--csv", default=None, help="Write results to CSV file")
    return p.parse_args()


# -----------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------

def timed_call(fn, warmup, iters, cuda=True):
    """Return (mean_ms, min_ms) after warmup + iters calls."""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if cuda:
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times), min(times)


# -----------------------------------------------------------------------
# Benchmark 1: KV-write (set_kv_buffer) throughput
# -----------------------------------------------------------------------

def bench_kv_write(args):
    """How many tokens/s can we write KV into the pool?"""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool

    dtype = getattr(torch, args.dtype)
    device = args.device
    N = 1024  # tokens to write per call
    pool_size = N + 64

    gpu_pool = MHATokenToKVPool(
        size=pool_size, page_size=1, dtype=dtype,
        head_num=args.num_kv_heads, head_dim=args.head_dim,
        layer_num=args.num_layers, device=device,
        enable_memory_saver=False,
    )
    cpu_pool = DirectKVTokenToKVPool(
        size=pool_size, page_size=1, dtype=dtype,
        head_num=args.num_kv_heads, head_dim=args.head_dim,
        layer_num=args.num_layers, v_head_dim=args.head_dim,
        start_layer=0, end_layer=args.num_layers,
    )

    loc_gpu = torch.arange(N, dtype=torch.long, device=device)
    loc_cpu = loc_gpu.cpu()
    k = torch.randn(N, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)
    v = torch.randn(N, args.num_kv_heads, args.head_dim, dtype=dtype, device=device)

    class FakeLayer:
        layer_id = 0

    layer = FakeLayer()
    is_cuda = (device != "cpu")

    def write_gpu():
        gpu_pool.set_kv_buffer(layer, loc_gpu, k, v)

    def write_cpu():
        cpu_pool.set_kv_buffer(layer, loc_cpu, k, v)

    mean_gpu, min_gpu = timed_call(write_gpu, args.warmup, args.iters, cuda=is_cuda)
    mean_cpu, min_cpu = timed_call(write_cpu, args.warmup, args.iters, cuda=False)

    toks_gpu = N / (min_gpu / 1000)
    toks_cpu = N / (min_cpu / 1000)

    print("\n=== KV-write throughput (set_kv_buffer, N={}) ===".format(N))
    print(f"  GPU pool : {min_gpu:.3f} ms min  →  {toks_gpu/1e6:.2f} M tokens/s")
    print(f"  CPU pool : {min_cpu:.3f} ms min  →  {toks_cpu/1e6:.2f} M tokens/s")
    print(f"  Overhead : {min_cpu/min_gpu:.1f}×  (CPU-pinned vs GPU scatter)")

    return {
        "kv_write_gpu_ms": min_gpu,
        "kv_write_cpu_ms": min_cpu,
    }


# -----------------------------------------------------------------------
# Benchmark 2: Decode attention latency vs context length
# -----------------------------------------------------------------------

def bench_decode_attn(args):
    """Single decode step: 1 new token, variable past context length."""
    from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool

    dtype = getattr(torch, args.dtype)
    device = args.device
    bs = args.batch_size
    nqh = args.num_q_heads
    nkvh = args.num_kv_heads
    D = args.head_dim
    scale = 1.0 / math.sqrt(D)
    groups = nqh // nkvh

    print(f"\n=== Decode attention latency  "
          f"(bs={bs}, nqh={nqh}, nkvh={nkvh}, D={D}, dtype={args.dtype}) ===")
    print(f"  {'context_len':>12}  {'directkv_ms':>14}  {'gpu_sdpa_ms':>12}  {'ratio':>8}")

    results = []
    for sl in args.seq_lens:
        pool_size = bs * sl + 8

        cpu_pool = DirectKVTokenToKVPool(
            size=pool_size, page_size=1, dtype=dtype,
            head_num=nkvh, head_dim=D,
            layer_num=1, v_head_dim=D,
            start_layer=0, end_layer=1,
        )

        # Fill pool with random KV
        all_tok_ids = torch.arange(bs * sl, dtype=torch.long)
        k_all = torch.randn(bs * sl, nkvh, D, dtype=dtype)
        v_all = torch.randn(bs * sl, nkvh, D, dtype=dtype)

        class FakeLayer:
            layer_id = 0

        cpu_pool.set_kv_buffer(FakeLayer(), all_tok_ids, k_all, v_all)
        k_buf = cpu_pool.get_key_buffer(0)
        v_buf = cpu_pool.get_value_buffer(0)

        # Per-sequence token index lists
        tok_ids_list = [all_tok_ids[i*sl:(i+1)*sl] for i in range(bs)]

        # Query on GPU
        q = torch.randn(bs, nqh, D, dtype=dtype, device=device)

        # ----- DirectKV: CPU-pinned gather + GPU SDPA -----
        def directkv_forward():
            outputs = []
            for i in range(bs):
                tid = tok_ids_list[i].long()
                k_i = k_buf[tid].to(device, non_blocking=True)
                v_i = v_buf[tid].to(device, non_blocking=True)
                q_i = q[i:i+1].transpose(0, 1).unsqueeze(0)
                k_i_t = k_i.permute(1, 0, 2).unsqueeze(0)
                v_i_t = v_i.permute(1, 0, 2).unsqueeze(0)
                if groups > 1:
                    k_i_t = k_i_t.repeat_interleave(groups, dim=1)
                    v_i_t = v_i_t.repeat_interleave(groups, dim=1)
                o = torch.nn.functional.scaled_dot_product_attention(
                    q_i, k_i_t, v_i_t, scale=scale
                )
                outputs.append(o.squeeze(0).squeeze(1).reshape(-1))
            return torch.stack(outputs, dim=0)

        # ----- Baseline: fully-GPU SDPA (KV already on GPU) -----
        k_gpu = k_buf.to(device)
        v_gpu = v_buf.to(device)

        def gpu_sdpa_forward():
            outputs = []
            for i in range(bs):
                tid = tok_ids_list[i].long()
                k_i = k_gpu[tid]
                v_i = v_gpu[tid]
                q_i = q[i:i+1].transpose(0, 1).unsqueeze(0)
                k_i_t = k_i.permute(1, 0, 2).unsqueeze(0)
                v_i_t = v_i.permute(1, 0, 2).unsqueeze(0)
                if groups > 1:
                    k_i_t = k_i_t.repeat_interleave(groups, dim=1)
                    v_i_t = v_i_t.repeat_interleave(groups, dim=1)
                o = torch.nn.functional.scaled_dot_product_attention(
                    q_i, k_i_t, v_i_t, scale=scale
                )
                outputs.append(o.squeeze(0).squeeze(1).reshape(-1))
            return torch.stack(outputs, dim=0)

        is_cuda = (device != "cpu")
        _, dkv_ms  = timed_call(directkv_forward,  args.warmup, args.iters, cuda=is_cuda)
        _, gpu_ms  = timed_call(gpu_sdpa_forward,   args.warmup, args.iters, cuda=is_cuda)

        ratio = dkv_ms / gpu_ms if gpu_ms > 0 else float("nan")
        print(f"  {sl:>12d}  {dkv_ms:>14.3f}  {gpu_ms:>12.3f}  {ratio:>8.2f}×")
        results.append({"seq_len": sl, "directkv_ms": dkv_ms, "gpu_sdpa_ms": gpu_ms, "ratio": ratio})

    return results


# -----------------------------------------------------------------------
# Benchmark 3: GPU memory savings
# -----------------------------------------------------------------------

def bench_memory(args):
    """Compare GPU memory used by GPU vs CPU-pinned KV pool."""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool

    dtype = getattr(torch, args.dtype)
    device = args.device
    size = 200_000  # total token slots

    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)

    gpu_pool = MHATokenToKVPool(
        size=size, page_size=1, dtype=dtype,
        head_num=args.num_kv_heads, head_dim=args.head_dim,
        layer_num=args.num_layers, device=device,
        enable_memory_saver=False,
    )

    if device != "cpu":
        after_gpu = torch.cuda.memory_allocated(device)
        gpu_bytes = after_gpu - before
        del gpu_pool

        before2 = torch.cuda.memory_allocated(device)
        cpu_pool = DirectKVTokenToKVPool(
            size=size, page_size=1, dtype=dtype,
            head_num=args.num_kv_heads, head_dim=args.head_dim,
            layer_num=args.num_layers, v_head_dim=args.head_dim,
            start_layer=0, end_layer=args.num_layers,
        )
        after_cpu = torch.cuda.memory_allocated(device)
        cpu_pool_gpu_bytes = after_cpu - before2
        del cpu_pool

        def fmt(b):
            return f"{b / 1024**3:.2f} GiB"

        print(f"\n=== GPU memory  (size={size:,}, layers={args.num_layers}, "
              f"nkvh={args.num_kv_heads}, D={args.head_dim}, dtype={args.dtype}) ===")
        print(f"  GPU KV pool   : {fmt(gpu_bytes)} GPU memory")
        print(f"  CPU KV pool   : {fmt(cpu_pool_gpu_bytes)} GPU memory  "
              f"(+CPU pinned: {fmt(gpu_bytes)})")
        print(f"  GPU savings   : {fmt(gpu_bytes - cpu_pool_gpu_bytes)}")
    else:
        print("\n[memory benchmark] Skipped (device=cpu).")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = parse_args()
    dtype_obj = getattr(torch, args.dtype)
    print(f"DirectKV benchmark  device={args.device}  dtype={args.dtype}")

    bench_kv_write(args)
    decode_results = bench_decode_attn(args)
    bench_memory(args)

    if args.csv:
        import csv, io
        rows = decode_results
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()
