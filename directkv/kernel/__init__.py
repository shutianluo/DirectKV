"""
DirectKV CUDA Extension Loader
================================
Lazy JIT-compilation of the kernel wrappers using torch.utils.cpp_extension.load.

Two extensions are provided:
  directkv_fwd  — qcentric_attn_kernel (original, Q-centric attention)
  smpv2_fwd     — sm_parallel_v2_cpu_kv_fwd_kernel (fused projection + RoPE + attention)

To use the custom kernel path:
  1. Set DIRECTKV_USE_CUSTOM_KERNEL=1
  2. Set CUTLASS_INCLUDE_DIR to the CUTLASS headers root, e.g.:
       /home/ubuntu/.local/lib/python3.10/site-packages/deep_gemm/include
  3. Set KERNEL_SRC_DIR to the autoresearch kernel sources (for directkv_fwd), e.g.:
       /workspace/sglang/flash-attention/autoresearch/kernel

The extensions are compiled on first use and cached by torch.
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Default search paths for CUTLASS and kernel sources.
# Checked in order; first existing directory wins.
def _site() -> str:
    try:
        import site
        return site.getusersitepackages()
    except Exception:
        return "/home/ubuntu/.local/lib/python3.10/site-packages"

_DEFAULT_CUTLASS_PATHS = [
    "/sgl-workspace/sglang/sgl-kernel/build/_deps/repo-cutlass-src/include",
    "/workspace/sglang/flash-attention/autoresearch/kernel/../../../sgl-kernel/build/_deps/repo-cutlass-src/include",
    # GH200 / deep_gemm bundled CUTLASS (SM90 capable)
    os.path.join(_site(), "deep_gemm", "include"),
    # flashinfer bundles CUTLASS too — available whenever sglang[all] is installed
    os.path.join(_site(), "flashinfer", "data", "cutlass", "include"),
]

_DEFAULT_KERNEL_SRC = "/workspace/sglang/flash-attention/autoresearch/kernel"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXT_SOURCE = os.path.join(_THIS_DIR, "directkv_extension.cu")

# SMPv2 extension — kernel sources are in kernel/ relative to the repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))   # DirectKV_AE-main/
_DEFAULT_SMPV2_KERNEL_SRC = os.path.join(_REPO_ROOT, "kernel")
_SMPV2_EXT_SOURCE = os.path.join(_THIS_DIR, "smpv2_extension.cu")


def _find_cutlass_include() -> Optional[str]:
    """Search common locations for CUTLASS include directory."""
    user_path = os.environ.get("CUTLASS_INCLUDE_DIR")
    if user_path:
        if os.path.isdir(user_path):
            return user_path
        raise RuntimeError(
            f"CUTLASS_INCLUDE_DIR is set to {user_path!r} but that directory "
            f"does not exist."
        )

    for p in _DEFAULT_CUTLASS_PATHS:
        if os.path.isdir(p):
            return p

    return None


@lru_cache(maxsize=1)
def _load_extension():
    """JIT-compile the directkv CUDA extension (once per process)."""
    cutlass_inc = _find_cutlass_include()
    if cutlass_inc is None:
        raise RuntimeError(
            "CUTLASS headers not found. Set CUTLASS_INCLUDE_DIR to your CUTLASS "
            "include directory (e.g., /path/to/cutlass/include)."
        )

    kernel_src = os.environ.get("DIRECTKV_KERNEL_SRC", _DEFAULT_KERNEL_SRC)
    if not os.path.isdir(kernel_src):
        raise RuntimeError(
            f"DirectKV kernel source directory not found: {kernel_src!r}. "
            f"Set DIRECTKV_KERNEL_SRC to the autoresearch/kernel directory."
        )

    if not os.path.isfile(_EXT_SOURCE):
        raise RuntimeError(
            f"directkv_extension.cu not found at {_EXT_SOURCE!r}. "
            f"Ensure the directkv_kernel package is correctly installed."
        )

    logger.info(
        f"[DirectKV] JIT-compiling directkv_extension.cu "
        f"(CUTLASS={cutlass_inc}, kernel_src={kernel_src}) ..."
    )

    nvcc_flags = [
        "-O3",
        "-arch=sm_90",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-DCUTE_ARCH_MMA_SM90A_ENABLED",
        "-DCUTLASS_ARCH_MMA_SM90_ENABLED",
        f"-I{cutlass_inc}",
        f"-I{kernel_src}",
    ]

    ext = torch.utils.cpp_extension.load(
        name="directkv_ext",
        sources=[_EXT_SOURCE],
        extra_cuda_cflags=nvcc_flags,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=bool(int(os.environ.get("DIRECTKV_VERBOSE_BUILD", "0"))),
    )
    logger.info("[DirectKV] Extension compiled successfully.")
    return ext


def directkv_fwd(
    Q: torch.Tensor,       # (B, S_new, num_q_heads, head_dim)  GPU
    Kpast: torch.Tensor,   # (B, S_past, num_kv_heads, head_dim) CPU-pinned
    Vpast: torch.Tensor,   # (B, S_past, num_kv_heads, v_head_dim) CPU-pinned
    Knew: torch.Tensor,    # (B, S_new, num_kv_heads, head_dim)  CPU-pinned
    Vnew: torch.Tensor,    # (B, S_new, num_kv_heads, v_head_dim) CPU-pinned
    softmax_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """
    Call the qcentric_attn_kernel via the JIT-compiled C extension.

    Returns O: (B, S_new, num_q_heads, head_dim) on GPU.
    """
    ext = _load_extension()
    return ext.directkv_fwd(Q, Kpast, Vpast, Knew, Vnew, softmax_scale, is_causal)


# ---------------------------------------------------------------------------
# SMPv2 CPU-KV extension  (fused projection + Neox RoPE + attention, SM90)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_smpv2_extension():
    """JIT-compile the smpv2_extension.cu (SM90, bf16, once per process)."""
    cutlass_inc = _find_cutlass_include()
    if cutlass_inc is None:
        raise RuntimeError(
            "CUTLASS headers not found. Set CUTLASS_INCLUDE_DIR to your CUTLASS "
            "include directory (e.g., /home/ubuntu/.local/lib/python3.10/"
            "site-packages/deep_gemm/include)."
        )

    kernel_src = os.environ.get("DIRECTKV_SMPV2_KERNEL_SRC", _DEFAULT_SMPV2_KERNEL_SRC)
    if not os.path.isdir(kernel_src):
        raise RuntimeError(
            f"SMPv2 kernel source directory not found: {kernel_src!r}. "
            f"Set DIRECTKV_SMPV2_KERNEL_SRC to the DirectKV kernel/ directory."
        )

    if not os.path.isfile(_SMPV2_EXT_SOURCE):
        raise RuntimeError(
            f"smpv2_extension.cu not found at {_SMPV2_EXT_SOURCE!r}."
        )

    logger.info(
        f"[DirectKV-SMPv2] JIT-compiling smpv2_extension.cu "
        f"(CUTLASS={cutlass_inc}, kernel_src={kernel_src}) ..."
    )

    # Locate pybind11 headers (needed on systems where torch doesn't bundle them)
    try:
        import pybind11
        pybind11_inc = pybind11.get_include()
    except ImportError:
        pybind11_inc = None

    # Do NOT define CUTE_ARCH_MMA_SM90A_ENABLED / CUTLASS_ARCH_MMA_SM90_ENABLED /
    # CUTE_ARCH_TMA_SM90_ENABLED here. CUTLASS auto-defines them during device
    # compilation (when __CUDA_ARCH__ >= 900). Defining them globally would
    # trigger CUTLASS_DEVICE synclog calls inside CUTE_HOST_DEVICE fma() bodies
    # during host compilation, causing nvcc errors.
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        f"-I{cutlass_inc}",
        f"-I{kernel_src}",
    ]

    cflags = ["-O3", "-std=c++17"]
    if pybind11_inc:
        cflags.append(f"-I{pybind11_inc}")
        nvcc_flags.append(f"-I{pybind11_inc}")

    # Force sm_90a only — prevents torch from adding generic compute_90 PTX gencode
    # flags that conflict with the SM90A WGMMA / synclog requirements.
    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        ext = torch.utils.cpp_extension.load(
            name="smpv2_ext",
            sources=[_SMPV2_EXT_SOURCE],
            extra_cuda_cflags=nvcc_flags,
            extra_cflags=cflags,
            extra_ldflags=["-lcuda"],
            verbose=bool(int(os.environ.get("DIRECTKV_VERBOSE_BUILD", "0"))),
        )
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
    logger.info("[DirectKV-SMPv2] Extension compiled successfully.")
    return ext


def smpv2_fwd(
    X: torch.Tensor,                       # [B, S_new, hidden_dim]      GPU bf16
    Wk: torch.Tensor,                      # [NH, head_dim, hidden_dim]  GPU bf16
    Wv: torch.Tensor,                      # [NH, head_dim, hidden_dim]  GPU bf16
    Q: torch.Tensor,                       # [B, S_new, NH, head_dim]    GPU bf16
    K_cpu: torch.Tensor,                   # [B, S_total, NH, head_dim]  CPU-pinned bf16
    V_cpu: torch.Tensor,                   # [B, S_total, NH, head_dim]  CPU-pinned bf16
    cos_sin: Optional[torch.Tensor],       # [max_pos, head_dim] GPU fp32, or None
    softmax_scale: float,
    is_causal: bool,
    seqlen_past: int,
    q_pos_start: int,
) -> torch.Tensor:
    """
    SMPv2 fused forward pass: X->K,V projection + Neox RoPE + attention.

    All new tokens are projected from X using Wk/Wv inside the SM90 kernel.
    RoPE rotation is applied in smem after projection when cos_sin is not None.
    Past KV tokens are loaded from CPU-pinned K_cpu/V_cpu via NVLink-C2C TMA.

    Constraints:
      - head_dim == 128
      - seqlen_past % 64 == 0,  S_new % 64 == 0
      - K_cpu.size(1) == seqlen_past + S_new  (S_total)

    Returns O: [B, S_new, NH, head_dim] GPU bf16.
    """
    ext = _load_smpv2_extension()
    return ext.smpv2_fwd(
        X, Wk, Wv, Q, K_cpu, V_cpu,
        cos_sin, softmax_scale, is_causal,
        seqlen_past, q_pos_start,
    )


def smpv2_init_tma(
    layer_id: int,
    X_max: torch.Tensor,     # [max_bs, S_new_max, H]          GPU bf16
    Wk: torch.Tensor,        # [NH, D, H]                      GPU bf16
    Wv: torch.Tensor,        # [NH, D, H]                      GPU bf16
    Q_max: torch.Tensor,     # [max_bs, S_new_max, NH, D]      GPU bf16
    K_max: torch.Tensor,     # [max_bs, S_total_max, NH, D]    CPU-pinned bf16
    V_max: torch.Tensor,     # [max_bs, S_total_max, NH, D]    CPU-pinned bf16
) -> None:
    """
    Pre-build TMA descriptors for one layer (CUDA graph init).

    Call once per layer during init_cuda_graph_state, before any graph capture.
    All tensor base pointers must remain stable (no reallocation) for the
    lifetime of the graph.

    Subsequent smpv2_fwd_graph calls for this layer_id use these descriptors.
    """
    ext = _load_smpv2_extension()
    ext.smpv2_init_tma(layer_id, X_max, Wk, Wv, Q_max, K_max, V_max)


def smpv2_fwd_graph(
    layer_id: int,
    X_pad: torch.Tensor,         # [max_bs, S_new, H]        GPU bf16 (slot filled, rest 0)
    Q_pad: torch.Tensor,         # [max_bs, S_new, NH, D]    GPU bf16 (slot filled, rest 0)
    O_out: torch.Tensor,         # [max_bs, S_new, NH, D]    GPU bf16 (pre-allocated output)
    n_past_dev: torch.Tensor,    # [1] int32 GPU — num_n_blocks_past for this step
    n_blks_dev: torch.Tensor,    # [1] int32 GPU — num_n_blocks for this step
    cos_sin: Optional[torch.Tensor],
    softmax_scale: float,
    is_causal: bool,
    S_total_max: int,
    seqlen_past_max: int,
    req_pool_indices: Optional[torch.Tensor] = None,  # [bs] int32 GPU: batch_pos → pool row
) -> torch.Tensor:
    """
    Graph-safe SMPv2 forward (skips TMA descriptor creation).

    n_past_dev and n_blks_dev are GPU int32 tensors updated by the host before
    each graph replay via tensor.fill_() or copy_(). The kernel reads them from
    device memory at execution time, allowing variable seqlen across replays.

    req_pool_indices: maps batch position to pool row (updated before each replay).
    Null = kernel uses blockIdx.y directly (non-graph path).

    Requires smpv2_init_tma(layer_id, ...) to have been called first.
    Returns O_out (same tensor) for convenient chaining.
    """
    ext = _load_smpv2_extension()
    return ext.smpv2_fwd_graph(
        layer_id, X_pad, Q_pad, O_out, n_past_dev, n_blks_dev,
        cos_sin, softmax_scale, is_causal,
        S_total_max, seqlen_past_max, req_pool_indices,
    )
