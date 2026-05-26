#!/usr/bin/env python3
"""
DirectKV Track B — End-to-End Performance Estimation
======================================================
Measures actual kernel latency for the SMPv2 fused path (Track B) and
compares it against Track A (SGLang-style: separate projection + SDPA).

Produces estimates for:
  Fig 10 — Per-token latency vs request rate (5–30 req/s)
  Fig 11 — Long-context latency and CPU memory savings

Run:
  CUTLASS_INCLUDE_DIR=/home/ubuntu/.local/lib/python3.10/site-packages/deep_gemm/include \\
  python benchmarks/bench_directkv_trackb_e2e.py

If the SMPv2 extension is not compiled, the script falls back to a PyTorch
simulation of the Track B kernel and emits estimated (not measured) timings.
"""

import math
import os
import sys
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

CONFIGS: Dict[str, dict] = {
    "LLaMA-3-8B": {
        "hidden_dim":   4096,
        "num_q_heads":  32,
        "num_kv_heads": 8,
        "head_dim":     128,
        "use_rope":     True,
        "rope_theta":   500_000.0,
        "max_pos":      8192,
        "layers":       32,
        "bytes_per_param": 2,  # bf16
    },
    "OPT-6.7B": {
        "hidden_dim":   4096,
        "num_q_heads":  32,
        "num_kv_heads": 32,
        "head_dim":     128,
        "use_rope":     False,
        "rope_theta":   None,
        "max_pos":      2048,
        "layers":       32,
        "bytes_per_param": 2,
    },
    "OPT-30B": {
        "hidden_dim":   7168,
        "num_q_heads":  56,
        "num_kv_heads": 56,
        "head_dim":     128,
        "use_rope":     False,
        "rope_theta":   None,
        "max_pos":      2048,
        "layers":       48,
        "bytes_per_param": 2,
    },
}

WARMUP  = 20
ITERS   = 100
DTYPE   = torch.bfloat16
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ShareGPT-like workload parameters (Fig 10)
REQUEST_RATES   = [5, 10, 15, 20, 25, 30]   # req/s
AVG_PREFILL_LEN = 512                         # tokens
AVG_DECODE_LEN  = 256                         # tokens
# Per-token service latency (ms) is measured at AVG_CONTEXT below
AVG_CONTEXT     = 512

# Sequence lengths for long-context sweep (Fig 11)
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192]

# GH200 NVLink-C2C bandwidth (GB/s)
NVLINK_BW = 380.0


# ---------------------------------------------------------------------------
# Load SMPv2 extension (best-effort)
# ---------------------------------------------------------------------------

_smpv2_fwd = None
_KERNEL_AVAILABLE = False

def _try_load_kernel():
    global _smpv2_fwd, _KERNEL_AVAILABLE
    cutlass = os.environ.get(
        "CUTLASS_INCLUDE_DIR",
        "/home/ubuntu/.local/lib/python3.10/site-packages/deep_gemm/include",
    )
    if not os.path.isdir(cutlass):
        print(f"[warn] CUTLASS not found at {cutlass}; using PyTorch simulation.")
        return
    try:
        import torch.utils.cpp_extension  # noqa: F401
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from sglang_ext.directkv_kernel import smpv2_fwd
        _smpv2_fwd = smpv2_fwd
        _KERNEL_AVAILABLE = True
        print("[info] SMPv2 kernel loaded successfully.")
    except Exception as e:
        print(f"[warn] Could not load SMPv2 kernel ({e}); using PyTorch simulation.")


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def bench(fn, warmup=WARMUP, iters=ITERS) -> float:
    """Returns mean latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(iters):
        fn()
    ev_e.record()
    torch.cuda.synchronize()
    return ev_s.elapsed_time(ev_e) / iters


# ---------------------------------------------------------------------------
# RoPE helper
# ---------------------------------------------------------------------------

def build_cos_sin(max_pos: int, head_dim: int, theta: float = 500_000.0) -> torch.Tensor:
    # +1 so position index max_pos is valid (e.g. new token at pos=S_past=max_pos)
    half  = head_dim // 2
    freq  = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t     = torch.arange(max_pos + 1, dtype=torch.float32)
    freqs = torch.outer(t, freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(DEVICE)


# ---------------------------------------------------------------------------
# Track A simulation (SGLang-style: projection + SDPA)
# ---------------------------------------------------------------------------

def track_a_decode(
    X: torch.Tensor,       # [1, 1, H]  GPU
    Wk: torch.Tensor,      # [NH_kv, D, H]  GPU
    Wv: torch.Tensor,      # [NH_kv, D, H]  GPU
    Wq: torch.Tensor,      # [NH_q, D, H]  GPU
    K_past: torch.Tensor,  # [1, S_past, NH_kv, D]  GPU
    V_past: torch.Tensor,  # [1, S_past, NH_kv, D]  GPU
    scale: float,
    cos_sin: Optional[torch.Tensor],
    pos: int,
) -> torch.Tensor:
    """Simulate SGLang Track A: separate QKV projection + SDPA."""
    H    = X.shape[-1]
    NH_q = Wq.shape[0]
    NH_k = Wk.shape[0]
    D    = Wk.shape[1]

    # Project Q, K, V (SGLang does these as separate GEMMs or fused QKV)
    X2  = X.view(1, H)
    Q   = (X2 @ Wq.view(NH_q * D, H).T).view(1, NH_q, D)   # [1, NH_q, D]
    K_n = (X2 @ Wk.view(NH_k * D, H).T).view(1, NH_k, D)
    V_n = (X2 @ Wv.view(NH_k * D, H).T).view(1, NH_k, D)

    # RoPE for Q and K
    if cos_sin is not None:
        half  = D // 2
        cs    = cos_sin[pos]                            # [D]
        cos_v = cs[:half]
        sin_v = cs[half:]
        def rot(x):
            lo, hi = x[..., :half].float(), x[..., half:].float()
            return torch.cat([lo * cos_v - hi * sin_v,
                              hi * cos_v + lo * sin_v], dim=-1).to(x.dtype)
        Q   = rot(Q)
        K_n = rot(K_n)

    # Concat K_new to past
    K_all = torch.cat([K_past.squeeze(0), K_n], dim=0)  # [S_total, NH_k, D]
    V_all = torch.cat([V_past.squeeze(0), V_n], dim=0)

    # SDPA
    nqh, nkv = NH_q, NH_k
    q_4d = Q.unsqueeze(0).transpose(0, 1).unsqueeze(2)             # [1, nqh, 1, D]
    k_4d = K_all.permute(1, 0, 2).unsqueeze(0)                     # [1, nkv, S_total, D]
    v_4d = V_all.permute(1, 0, 2).unsqueeze(0)
    if nqh != nkv:
        k_4d = k_4d.repeat_interleave(nqh // nkv, dim=1)
        v_4d = v_4d.repeat_interleave(nqh // nkv, dim=1)

    return F.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=scale)


# ---------------------------------------------------------------------------
# Track B simulation (Python projection, no kernel)
# ---------------------------------------------------------------------------

def track_b_python_decode(
    X: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
    Wq: torch.Tensor,
    K_past: torch.Tensor,
    V_past: torch.Tensor,
    scale: float,
    cos_sin: Optional[torch.Tensor],
    pos: int,
) -> torch.Tensor:
    """Track B decode path: Python projection (same ops as Track A but K,V bypass GPU pool)."""
    # In practice Track B avoids the GPU→CPU copy of K,V (writes directly to pinned)
    # For timing purposes, this path measures the projection + attention separately
    return track_a_decode(X, Wk, Wv, Wq, K_past, V_past, scale, cos_sin, pos)


# ---------------------------------------------------------------------------
# Track B kernel path (SMPv2 extension)
# ---------------------------------------------------------------------------

def track_b_kernel_decode(
    X: torch.Tensor,       # [1, 64, H]  GPU bf16 (padded)
    Wk: torch.Tensor,      # [NH_kv, D, H]  GPU bf16
    Wv: torch.Tensor,      # [NH_kv, D, H]  GPU bf16
    Q: torch.Tensor,       # [1, 64, NH_q, D]  GPU bf16 (padded)
    K_cpu: torch.Tensor,   # [1, S_total, NH_kv, D]  CPU-pinned bf16
    V_cpu: torch.Tensor,   # [1, S_total, NH_kv, D]  CPU-pinned bf16
    scale: float,
    cos_sin: Optional[torch.Tensor],
    seqlen_past: int,
) -> torch.Tensor:
    """Call smpv2_fwd with pre-padded inputs. q_pos_start=-1 (Q already RoPE'd)."""
    return _smpv2_fwd(
        X, Wk, Wv, Q, K_cpu, V_cpu,
        cos_sin, scale, True,     # is_causal
        seqlen_past, -1,          # q_pos_start=-1 → skip Q rotation
    )


# ---------------------------------------------------------------------------
# Per-component microbenchmark
# ---------------------------------------------------------------------------

def bench_components(cfg: dict, S_past: int) -> dict:
    """Measure individual components at a given past sequence length."""
    NH_q = cfg["num_q_heads"]
    NH_k = cfg["num_kv_heads"]
    D    = cfg["head_dim"]
    H    = cfg["hidden_dim"]
    use_rope = cfg["use_rope"]
    theta    = cfg.get("rope_theta", 500_000.0) or 500_000.0
    scale    = D ** -0.5

    X  = torch.randn(1, 1, H, dtype=DTYPE, device=DEVICE)
    Wq = torch.randn(NH_q, D, H, dtype=DTYPE, device=DEVICE)
    Wk = torch.randn(NH_k, D, H, dtype=DTYPE, device=DEVICE)
    Wv = torch.randn(NH_k, D, H, dtype=DTYPE, device=DEVICE)
    K_past_gpu = torch.randn(1, S_past, NH_k, D, dtype=DTYPE, device=DEVICE)
    V_past_gpu = torch.randn_like(K_past_gpu)
    cos_sin = build_cos_sin(cfg["max_pos"], D, theta) if use_rope else None

    # ── Track A: projection + SDPA ──────────────────────────────────────
    ms_a = bench(lambda: track_a_decode(
        X, Wk, Wv, Wq, K_past_gpu, V_past_gpu, scale, cos_sin, S_past
    ))

    # ── projection only (K + V GEMMs) ───────────────────────────────────
    X2   = X.view(1, H)
    Wkv2 = torch.cat([Wk, Wv], dim=0).view(2 * NH_k * D, H)
    ms_proj = bench(lambda: X2 @ Wkv2.T)

    # ── SDPA only ────────────────────────────────────────────────────────
    Q_gpu   = torch.randn(1, NH_q, 1, D, dtype=DTYPE, device=DEVICE)
    K_4d    = K_past_gpu.squeeze(0).permute(1, 0, 2).unsqueeze(0)
    V_4d    = V_past_gpu.squeeze(0).permute(1, 0, 2).unsqueeze(0)
    if NH_q != NH_k:
        K_4d = K_4d.repeat_interleave(NH_q // NH_k, dim=1)
        V_4d = V_4d.repeat_interleave(NH_q // NH_k, dim=1)
    ms_sdpa = bench(lambda: F.scaled_dot_product_attention(Q_gpu, K_4d, V_4d, scale=scale))

    # NVLink-C2C KV load model
    kBN = 64
    tiles  = math.ceil(S_past / kBN)
    t_bytes = kBN * D * 2 * 2              # K + V, bf16
    nvl_ms = (tiles * t_bytes / 1e9) / NVLINK_BW * 1e3

    # ── Track B kernel (if available and seqlen aligned) ─────────────────
    ms_b_kernel = None
    if _KERNEL_AVAILABLE and NH_q == NH_k and S_past % 64 == 0 and S_past > 0:
        S_pad   = 64
        S_total = S_past + S_pad
        X_pad   = torch.zeros(1, S_pad, H, dtype=DTYPE, device=DEVICE)
        X_pad[0, 0] = X[0, 0]
        Q_pad = torch.zeros(1, S_pad, NH_k, D, dtype=DTYPE, device=DEVICE)
        Q_pad[0, 0] = (X[0, 0] @ Wq[0].T).to(DTYPE)      # dummy Q head 0

        K_cpu = torch.zeros(1, S_total, NH_k, D, dtype=DTYPE, pin_memory=True)
        V_cpu = torch.zeros(1, S_total, NH_k, D, dtype=DTYPE, pin_memory=True)
        K_cpu[0, :S_past] = K_past_gpu.cpu().squeeze(0)
        V_cpu[0, :S_past] = V_past_gpu.cpu().squeeze(0)

        try:
            ms_b_kernel = bench(lambda: track_b_kernel_decode(
                X_pad, Wk, Wv, Q_pad, K_cpu, V_cpu, scale, cos_sin, S_past
            ))
        except Exception as e:
            print(f"    [kernel error at S_past={S_past}]: {e}")

    return {
        "ms_a": ms_a,
        "ms_proj": ms_proj,
        "ms_sdpa": ms_sdpa,
        "nvl_ms": nvl_ms,
        "ms_b_kernel": ms_b_kernel,
    }


# ---------------------------------------------------------------------------
# Memory savings (Fig 11 style)
# ---------------------------------------------------------------------------

def memory_savings_gb(cfg: dict, num_tokens: int) -> Tuple[float, float]:
    """
    Returns (gpu_kv_bytes, cpu_kv_bytes) for num_tokens stored in the KV pool.
    GPU baseline: MHATokenToKVPool stores per-layer KV on GPU.
    DirectKV:  CPU-pinned pool (no GPU KV memory).
    """
    layers = cfg["layers"]
    NH_k   = cfg["num_kv_heads"]
    D      = cfg["head_dim"]
    bpp    = cfg["bytes_per_param"]

    kv_per_token = layers * 2 * NH_k * D * bpp    # K + V, all layers
    return num_tokens * kv_per_token / 1e9         # GB


# ---------------------------------------------------------------------------
# Fig 10: Attention contribution to decode latency
# ---------------------------------------------------------------------------

def estimate_fig10(model_name: str, cfg: dict, ms_attn_per_layer: float,
                   ms_b_est: float) -> None:
    """
    Show per-layer decode attention breakdown for Track A vs Track B.

    NOTE: The actual Fig 10 (per-token latency vs request rate) requires a
    running SGLang server. Use bench_offload_serving.py for server-level data.
    This table shows the attention-layer savings that Drive the Track B speedup.
    """
    layers = cfg["layers"]
    # Approximate full-model token time: 32-48 layers of attention + 2× MLP
    # At batch_size=8 (typical decode batch), amortise: roughly /4 vs single-req
    MLP_FACTOR = 2.0
    total_attn_ms = ms_attn_per_layer * layers
    total_b_ms    = ms_b_est * layers
    model_ms_full = total_attn_ms * (1 + MLP_FACTOR)   # attention + MLP, batch=1
    model_b_full  = total_b_ms    * (1 + MLP_FACTOR)

    print(f"\n  {model_name} — Track A vs B projection  "
          f"({layers} layers, MLP≈{MLP_FACTOR}× attn)")
    print(f"  {'Path':>12} | {'Attn/layer':>12} | {'Attn×L(ms)':>12} | "
          f"{'Full-model est.':>16} | {'Speedup':>8}")
    print("  " + "-" * 68)
    print(f"  {'Track A':>12} | {ms_attn_per_layer:>12.3f} | {total_attn_ms:>12.2f} | "
          f"{model_ms_full:>16.0f} | {'1.00×':>8}")
    print(f"  {'Track B (est)':>12} | {ms_b_est:>12.3f} | {total_b_ms:>12.2f} | "
          f"{model_b_full:>16.0f} | {model_ms_full/model_b_full:>7.2f}×")
    print(f"\n  → Eliminates K+V projection ({ms_attn_per_layer - ms_b_est:.3f}ms/layer) "
          f"= {(total_attn_ms - total_b_ms):.1f}ms/token across all layers.")
    print(f"  → Full Fig 10 server data: run bench_offload_serving.py with "
          f"--attention-backend directkv-smpv2")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _try_load_kernel()

    print("=" * 72)
    print("DirectKV Track B — End-to-End Performance Estimation")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Kernel: {'AVAILABLE' if _KERNEL_AVAILABLE else 'PyTorch simulation'}")
    print(f"Dtype: {DTYPE} | NVLink-C2C model: {NVLINK_BW} GB/s")
    print("=" * 72)

    for model_name, cfg in CONFIGS.items():
        NH_q = cfg["num_q_heads"]
        NH_k = cfg["num_kv_heads"]
        D    = cfg["head_dim"]
        H    = cfg["hidden_dim"]
        use_rope = cfg["use_rope"]

        print(f"\n{'─' * 68}")
        print(f"Model: {model_name}  (NH_q={NH_q}, NH_kv={NH_k}, D={D}, H={H}, RoPE={use_rope})")
        print(f"{'─' * 68}")

        # Header
        kernel_col = " | B-kernel(ms)" if _KERNEL_AVAILABLE else ""
        print(f"  {'S_past':>8} | {'A-total(ms)':>12} | {'proj(ms)':>10} | "
              f"{'sdpa(ms)':>10} | {'nvlink(ms)':>11} | "
              f"{'B-overhead%':>12}{kernel_col}")
        print("  " + "-" * (80 + (15 if _KERNEL_AVAILABLE else 0)))

        ms_at_avg = None
        for S in SEQ_LENS:
            if S > cfg["max_pos"]:
                continue
            r = bench_components(cfg, S)

            ms_a       = r["ms_a"]
            ms_proj    = r["ms_proj"]
            ms_sdpa    = r["ms_sdpa"]
            nvl_ms     = r["nvl_ms"]
            ms_b_k     = r["ms_b_kernel"]

            # Track B overhead vs Track A: rope + projection hidden inside kernel
            # Estimated Track B: max(nvlink_load, sdpa) — projection overlapped
            ms_b_est   = max(nvl_ms, ms_sdpa)
            overhead   = (ms_b_est / ms_a - 1.0) * 100.0 if ms_a > 0 else 0.0

            if ms_b_k is not None:
                kernel_str = f" | {ms_b_k:>13.3f}"
            elif _KERNEL_AVAILABLE:
                kernel_str = f" | {'n/a':>13}"
            else:
                kernel_str = ""
            print(f"  {S:>8} | {ms_a:>12.3f} | {ms_proj:>10.3f} | "
                  f"{ms_sdpa:>10.3f} | {nvl_ms:>11.4f} | "
                  f"{overhead:>11.1f}%{kernel_str}")

            if S == AVG_CONTEXT:
                ms_at_avg = ms_a

        # ── Fig 10 projection ──────────────────────────────────────────────
        if ms_at_avg is not None:
            r_avg = bench_components(cfg, AVG_CONTEXT)
            ms_b_avg = r_avg["ms_b_kernel"] or max(r_avg["nvl_ms"], r_avg["ms_sdpa"])
            estimate_fig10(model_name, cfg, ms_at_avg, ms_b_avg)

    # ── Fig 11: Memory savings table ─────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("Fig 11 — CPU KV pool memory savings vs GPU baseline")
    print(f"{'Model':>20} | {'Tokens':>10} | {'GPU KV (GB)':>12} | {'CPU KV (GB)':>12} | {'Saved (%)':>10}")
    print("-" * 72)
    for model_name, cfg in CONFIGS.items():
        for num_toks in [10_000, 50_000, 100_000]:
            gpu_gb = memory_savings_gb(cfg, num_toks)
            # DirectKV: 0 GPU KV memory (all in CPU-pinned)
            cpu_gb = gpu_gb
            print(f"  {model_name:>18} | {num_toks:>10,} | {gpu_gb:>12.2f} | "
                  f"{cpu_gb:>12.2f} | {'100.0':>10}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("Track A vs Track B — Summary (LLaMA-3-8B, S_past=512)")
    print(f"{'=' * 72}")
    cfg = CONFIGS["LLaMA-3-8B"]
    r   = bench_components(cfg, 512)
    ms_a = r["ms_a"]
    nvl  = r["nvl_ms"]
    sdpa = r["ms_sdpa"]
    proj = r["ms_proj"]
    ms_b_est = max(nvl, sdpa)
    print(f"  Track A  = proj({proj:.3f}ms) + sdpa({sdpa:.3f}ms) = {ms_a:.3f} ms")
    print(f"  Track B  ≈ max(nvlink({nvl:.4f}ms), sdpa({sdpa:.3f}ms)) = {ms_b_est:.3f} ms")
    print(f"  Speedup  ≈ {ms_a / ms_b_est:.2f}×  (projection hidden behind NVLink load)")
    if r["ms_b_kernel"] is not None:
        print(f"  Track B kernel (measured) = {r['ms_b_kernel']:.3f} ms")
    print(f"\n  Eliminated: K+V projection GEMM ({proj:.3f} ms) per decode step per layer")
    print(f"  Memory:     0 GPU KV memory; all {memory_savings_gb(cfg, 100_000):.1f} GB in CPU-pinned")

    print(f"\n{'=' * 72}")
    print("Estimation complete.")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
