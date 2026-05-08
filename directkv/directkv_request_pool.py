"""
DirectKV per-request contiguous CPU-pinned KV pool.

Each active request gets its own row in a shared pre-allocated
  [max_reqs, max_seqlen_pad, NH_kv, head_dim]  CPU-pinned bf16 tensor per layer.

Layout aligns with directkv_fwd requirements:
  K_cpu / V_cpu : [1, S_total, NH_kv, D]  (view of pool row, no copy)
  S_total must be a multiple of 64 (kernel constraint).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_ALIGN = 64   # kernel tile size


def _pad64(n: int) -> int:
    return math.ceil(n / _ALIGN) * _ALIGN


class DirectKVRequestPool:
    """
    Shared pre-allocated CPU-pinned KV pool with per-request row assignment.

    One pool instance is created per layer. The pool shape is fixed at init:
      [max_reqs, max_seqlen_pad, NH_kv, head_dim]  CPU-pinned bf16

    Requests are assigned to rows via a free-list. The kernel can access
    any row as a [1, S_total, NH_kv, head_dim] view without any memory copy.
    """

    def __init__(
        self,
        max_reqs: int,
        max_seqlen: int,           # raw; padded up to multiple of 64 internally
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        start_layer: int = 0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.max_reqs       = max_reqs
        self.max_seqlen_pad = _pad64(max_seqlen)
        self.num_kv_heads   = num_kv_heads
        self.head_dim       = head_dim
        self.num_layers     = num_layers
        self.start_layer    = start_layer
        self.dtype          = dtype

        row_bytes = self.max_seqlen_pad * num_kv_heads * head_dim * dtype.itemsize
        total_gb  = 2 * max_reqs * num_layers * row_bytes / 1024**3
        logger.info(
            f"[DirectKVPool] Allocating {max_reqs} req × {num_layers} layers × "
            f"2 (K+V) × [{self.max_seqlen_pad}, {num_kv_heads}, {head_dim}] "
            f"CPU-pinned bf16: {total_gb:.2f} GB"
        )

        # k_pool[li][row] = Tensor[max_seqlen_pad, NH_kv, D]  CPU-pinned
        self.k_pool: List[torch.Tensor] = []
        self.v_pool: List[torch.Tensor] = []
        for _ in range(num_layers):
            k = torch.zeros(
                max_reqs, self.max_seqlen_pad, num_kv_heads, head_dim,
                dtype=dtype, pin_memory=True,
            )
            v = torch.zeros_like(k)
            self.k_pool.append(k)
            self.v_pool.append(v)

        # Free-list of row indices
        self._free: List[int] = list(range(max_reqs))
        self._req_to_row: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def alloc(self, req_id: int) -> None:
        if req_id in self._req_to_row:
            return   # already allocated (idempotent)
        if not self._free:
            raise RuntimeError(
                f"[DirectKVPool] Out of rows (max_reqs={self.max_reqs}). "
                "Increase max_reqs or reduce --max-running-requests."
            )
        row = self._free.pop()
        self._req_to_row[req_id] = row
        # Zero out the row so stale KV from a previous request doesn't leak.
        for li in range(self.num_layers):
            self.k_pool[li][row].zero_()
            self.v_pool[li][row].zero_()

    def free(self, req_id: int) -> None:
        row = self._req_to_row.pop(req_id, None)
        if row is not None:
            self._free.append(row)

    # ------------------------------------------------------------------
    # Per-layer tensor access — O(1), no memory copy
    # ------------------------------------------------------------------

    def get_kv(
        self, req_id: int, layer_id: int, s_total: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (K_view, V_view) shaped [1, s_total, NH_kv, D] CPU-pinned.
        s_total must be a multiple of 64 and <= max_seqlen_pad.
        """
        row = self._req_to_row[req_id]
        li  = layer_id - self.start_layer
        K = self.k_pool[li][row, :s_total].unsqueeze(0)   # [1, s_total, NH, D]
        V = self.v_pool[li][row, :s_total].unsqueeze(0)
        return K, V

    def get_row_tensor(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the full pool tensors for one layer (for CUDA graph max-shape descriptors)."""
        li = layer_id - self.start_layer
        return self.k_pool[li], self.v_pool[li]

    def row_of(self, req_id: int) -> int:
        return self._req_to_row[req_id]

    # ------------------------------------------------------------------
    # Batch view helper — used for bucketed decode (§5.5)
    # ------------------------------------------------------------------

    def get_batch_kv(
        self,
        req_ids: List[int],
        layer_id: int,
        s_total: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (K_batch, V_batch) shaped [len(req_ids), s_total, NH_kv, D].
        Rows are not necessarily contiguous in the pool, so this returns a
        stacked view (zero-copy only when rows happen to be contiguous).
        """
        rows = [self._req_to_row[r] for r in req_ids]
        li   = layer_id - self.start_layer
        K = self.k_pool[li][rows, :s_total]   # [B, s_total, NH, D]
        V = self.v_pool[li][rows, :s_total]
        return K, V

    # ------------------------------------------------------------------
    # CUDA graph support: static max-shape views
    # ------------------------------------------------------------------

    def get_cuda_graph_kv(
        self, layer_id: int, max_bs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (K, V) shaped [max_bs, max_seqlen_pad, NH_kv, D].
        Pointers are stable across decode steps — safe for TMA descriptor capture.
        """
        li = layer_id - self.start_layer
        return self.k_pool[li][:max_bs], self.v_pool[li][:max_bs]

    def __repr__(self) -> str:
        return (
            f"DirectKVRequestPool(max_reqs={self.max_reqs}, "
            f"max_seqlen_pad={self.max_seqlen_pad}, "
            f"layers={self.num_layers}, "
            f"free={len(self._free)})"
        )
