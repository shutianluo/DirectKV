#!/usr/bin/env python3
"""
Performance estimation for Track B (fused projection + RoPE in SMPv2 kernel).

Measures:
  - RoPE rotation overhead vs total attention time (PyTorch simulation)
  - Per-token latency scaling with sequence length
  - GPU memory transfer cost (NVLink-C2C model)
  - Throughput comparison: Track A (pre-projected) vs Track B (fused)

Run: python benchmarks/bench_smpv2_perf.py
"""

import math, time, sys
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIGS = {
    "LLaMA-3-8B": {
        "hidden_dim": 4096, "num_q_heads": 32, "num_kv_heads": 8,
        "head_dim": 128, "rope_theta": 500000.0, "use_rope": True,
        "max_pos": 8192,
    },
    "OPT-6.7B": {
        "hidden_dim": 4096, "num_q_heads": 32, "num_kv_heads": 32,
        "head_dim": 128, "rope_theta": None, "use_rope": False,
        "max_pos": 2048,
    },
    "OPT-30B": {
        "hidden_dim": 7168, "num_q_heads": 56, "num_kv_heads": 56,
        "head_dim": 128, "rope_theta": None, "use_rope": False,
        "max_pos": 2048,
    },
}

WARMUP_ITERS = 20
BENCH_ITERS = 100
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GH200 NVLink-C2C bandwidth (GB/s) — from kernel comments
NVLINK_C2C_BW_GBs = 380.0   # peak bidirectional
# Per-SM NVLink share for 1 CTA
KV_TILE_BYTES = 32 * 1024   # 16KB K + 16KB V per tile (kBlockN=64, D=128, bf16)

# Sequence lengths to sweep
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192]


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def bench_cuda_fn(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Run fn WARMUP_ITERS times then measure iters repetitions. Returns ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


# ---------------------------------------------------------------------------
# Reference functions
# ---------------------------------------------------------------------------

def build_cos_sin(max_pos, head_dim, theta=500000.0):
    half_D = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_D, dtype=torch.float32) / half_D))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(DEVICE)


def rope_rotate_batch(K, cos_sin, pos_start):
    """Rotate K [B, S, NH, D] at positions [pos_start, pos_start+S)."""
    B, S, NH, D = K.shape
    half_D = D // 2
    positions = torch.arange(pos_start, pos_start + S, dtype=torch.long, device=K.device)
    cs = cos_sin[positions]              # [S, D]
    cos = cs[:, :half_D].unsqueeze(0).unsqueeze(2)   # [1, S, 1, half_D]
    sin = cs[:, half_D:].unsqueeze(0).unsqueeze(2)
    k_f = K.float()
    k_lo, k_hi = k_f[..., :half_D], k_f[..., half_D:]
    K_rot = torch.cat([k_lo * cos - k_hi * sin,
                        k_hi * cos + k_lo * sin], dim=-1)
    return K_rot.to(K.dtype)


def attention_sdpa(Q, K, V, scale):
    """Q,K,V: [B*NH, S, D]. Returns [B*NH, S_q, D]."""
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)


# ---------------------------------------------------------------------------
# Microbenchmarks
# ---------------------------------------------------------------------------

def bench_rope_rotation(B, NH, S, D, device):
    """Time for RoPE rotation of a [B, S, NH, D] BF16 tensor."""
    K = torch.randn(B, S, NH, D, dtype=DTYPE, device=device)
    cos_sin = build_cos_sin(S + 100, D)

    def fn():
        rope_rotate_batch(K, cos_sin, 0)

    ms = bench_cuda_fn(fn)
    bytes_rw = B * S * NH * D * 2 * 2  # read + write, 2B per BF16 element
    bw_gbs = (bytes_rw / ms / 1e6)      # GB/s
    return ms, bw_gbs


def bench_qk_gemm(B, NH, S_q, S_kv, D, device):
    """Time for QK GEMM (batch scaled_dot_product_attention, no flash)."""
    Q = torch.randn(B * NH, S_q, D, dtype=DTYPE, device=device)
    K = torch.randn(B * NH, S_kv, D, dtype=DTYPE, device=device)
    V = torch.randn(B * NH, S_kv, D, dtype=DTYPE, device=device)
    scale = D ** -0.5

    def fn():
        attention_sdpa(Q, K, V, scale)

    ms = bench_cuda_fn(fn)
    return ms


def estimate_nvlink_load_ms(S_kv, NH, D, num_heads_per_block=1):
    """Estimate NVLink-C2C KV load latency for all KV tiles."""
    # Each tile: kBlockN=64 tokens × NH_per_block=1 head × D=128 × 2B × 2 (K+V)
    kBlockN = 64
    num_tiles = math.ceil(S_kv / kBlockN) * NH  # total tiles across all heads
    # But tiles are processed in parallel across heads (grid over heads)
    # → effective serial tiles per head = ceil(S_kv / kBlockN)
    tiles_serial = math.ceil(S_kv / kBlockN)
    tile_bytes = kBlockN * D * 2 * 2  # K + V, bf16
    total_bytes_per_head = tiles_serial * tile_bytes
    load_ms = (total_bytes_per_head / 1e9) / NVLINK_C2C_BW_GBs * 1e3
    return load_ms


def bench_projection(B, NH, S_new, D, H, device):
    """Time for X→K projection: X[B,S,H] @ Wk[NH,D,H].T → [B,S,NH,D]."""
    X = torch.randn(B, S_new, H, dtype=DTYPE, device=device)
    Wk = torch.randn(NH * D, H, dtype=DTYPE, device=device)

    def fn():
        # Simulate: X[B,S,H] @ Wk.T[H, NH*D] → reshape
        out = X.reshape(B * S_new, H) @ Wk.T
        out.reshape(B, S_new, NH, D)

    ms = bench_cuda_fn(fn)
    return ms


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis():
    print(f"{'='*70}")
    print("SMPv2 Track B Performance Estimation")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {DTYPE}  |  NVLink BW model: {NVLINK_C2C_BW_GBs} GB/s")
    print(f"{'='*70}\n")

    for model_name, cfg in CONFIGS.items():
        NH_q = cfg["num_q_heads"]
        NH_kv = cfg["num_kv_heads"]
        D = cfg["head_dim"]
        H = cfg["hidden_dim"]
        use_rope = cfg["use_rope"]
        B = 4   # batch size = 4 (typical decode batch)

        print(f"\n{'─'*60}")
        print(f"Model: {model_name}  (NH_q={NH_q}, NH_kv={NH_kv}, D={D}, H={H})")
        print(f"{'─'*60}")
        print(f"{'SeqLen':>8} | {'Attn(ms)':>9} | {'Rope(ms)':>9} | {'Proj(ms)':>9} | "
              f"{'NVLink(ms)':>10} | {'RopeOvhd%':>10} | {'TotEst(ms)':>10}")
        print("-" * 77)

        for S in SEQ_LENS:
            # 1. Attention (Q-inner, GPU side)
            attn_ms = bench_qk_gemm(B, NH_q, 1, S, D, DEVICE)

            # 2. RoPE rotation of K tile (per new-token tile, kBlockN=64)
            if use_rope:
                kBN = 64
                rope_ms, rope_bw = bench_rope_rotation(1, NH_kv, kBN, D, DEVICE)
                num_new_tiles = max(1, math.ceil(1 / kBN))   # decode: 1 new token → 1 tile
                total_rope_ms = rope_ms * num_new_tiles       # negligible for decode
                # For prefill with S_new tokens:
                rope_overhead_pct = (total_rope_ms / (attn_ms + 1e-9)) * 100
            else:
                total_rope_ms = 0.0
                rope_overhead_pct = 0.0

            # 3. X→K projection (Track B fuses this into the kernel)
            proj_ms = bench_projection(B, NH_kv, 1, D, H, DEVICE)  # 1 new token decode

            # 4. NVLink-C2C KV load model (all past tokens)
            nvlink_ms = estimate_nvlink_load_ms(S, NH_kv, D)

            # 5. Total estimate: max(NVLink_load, attn) + rope + proj (overlapped)
            # In the kernel: NVLink loads and Q-inner attention overlap partially
            # RoPE runs once per new-token tile (small, amortised over all attention)
            total_est_ms = max(attn_ms, nvlink_ms) + total_rope_ms

            rope_str = f"{total_rope_ms:.3f}" if use_rope else "—"
            proj_str = f"{proj_ms:.3f}"
            rope_pct = f"{rope_overhead_pct:.1f}%" if use_rope else "—"

            print(f"{S:>8} | {attn_ms:>9.3f} | {rope_str:>9} | {proj_str:>9} | "
                  f"{nvlink_ms:>10.4f} | {rope_pct:>10} | {total_est_ms:>10.3f}")

    # ── Summary: RoPE rotation BW utilisation ─────────────────────────────────
    print(f"\n{'='*70}")
    print("RoPE Rotation Bandwidth Utilisation (kBlockN=64, D=128, BF16)")
    print(f"{'='*70}")
    kBN, D = 64, 128
    ms, bw = bench_rope_rotation(1, 8, kBN, D, DEVICE)
    print(f"  Tile rotation latency : {ms*1000:.1f} µs")
    print(f"  Effective smem BW     : {bw:.1f} GB/s")
    print(f"  cos/sin HBM reads     : {kBN * D * 4 / 1024:.1f} KB per tile")
    tile_lat_us = ms * 1000
    qk_ref_ms = bench_qk_gemm(1, 8, 1, 256, D, DEVICE)
    print(f"  QK GEMM (S=256 ref)   : {qk_ref_ms*1000:.1f} µs")
    overhead_pct = tile_lat_us / (qk_ref_ms * 1000) * 100
    print(f"  Rotation overhead/QK  : {overhead_pct:.2f}%")

    # ── Track A vs Track B comparison ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Track A vs Track B: Time breakdown for LLaMA-3-8B decode, S=1024")
    print(f"{'='*70}")
    cfg = CONFIGS["LLaMA-3-8B"]
    NH_q, NH_kv, D, H = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["hidden_dim"]
    B, S = 4, 1024
    attn_ms = bench_qk_gemm(B, NH_q, 1, S, D, DEVICE)
    rope_ms_kv, _ = bench_rope_rotation(B, NH_kv, 64, D, DEVICE)  # per tile
    proj_ms = bench_projection(B, NH_kv, 1, D, H, DEVICE)
    nvl_ms = estimate_nvlink_load_ms(S, NH_kv, D)

    print(f"  Attention (QK+PV GEMM): {attn_ms:.3f} ms")
    print(f"  K RoPE rotation/tile  : {rope_ms_kv*1000:.1f} µs (1 tile for decode)")
    print(f"  X→K projection        : {proj_ms:.3f} ms")
    print(f"  NVLink-C2C KV load    : {nvl_ms:.4f} ms")
    print(f"")
    print(f"  Track A total est.    : {attn_ms:.3f} ms  (sglang proj + attention)")
    print(f"  Track B total est.    : {max(attn_ms, nvl_ms) + rope_ms_kv:.3f} ms  (kernel proj+rope+attn)")
    print(f"  Track B overhead      : ~{rope_ms_kv*1000:.1f} µs per new-token tile")

    print(f"\n{'='*70}")
    print("Estimation complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_analysis()
