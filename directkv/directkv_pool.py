"""
DirectKV CPU-pinned KV Cache Pool
==================================
Stores K and V cache tensors in **host-pinned (page-locked) CPU memory** so
that the GPU can access them directly via PCIe/NVLink without an explicit
device-to-device copy.  The logical token-index layout is identical to
MHATokenToKVPool, which lets the rest of the SGLang scheduler (ReqToTokenPool,
allocator, etc.) work unchanged.

Assumptions / MVP constraints
------------------------------
* BF16 / FP16 only (no FP8, no INT4).
* Causal MHA / GQA (no MLA, no sliding-window).
* No prefix/radix cache (caller must pass --disable-radix-cache).
* No speculative decoding.
* `enable_memory_saver` is silently ignored (pinned memory does not benefit
  from the lazy-allocation trick used for GPU memory).

Memory layout
-------------
Per layer:
  k_buffer[layer_idx]  shape = (size + 1, head_num, head_dim)   CPU-pinned
  v_buffer[layer_idx]  shape = (size + 1, head_num, v_head_dim) CPU-pinned

The "+1" sentinel matches MHATokenToKVPool convention (index 0 is reserved /
padding).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class DirectKVTokenToKVPool(KVCache):
    """
    CPU-pinned drop-in replacement for MHATokenToKVPool.

    All tensors live in host-pinned memory so the GPU can read them via
    unified virtual addressing without staging through an explicit cudaMemcpy.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        # KVCache base requires device & enable_memory_saver; we fix them here.
        device: str = "cpu",  # always CPU for pinned memory
        enable_memory_saver: bool = False,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        # Validate dtype before calling super().__init__ (which may log sizes)
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"DirectKVTokenToKVPool only supports float16 / bfloat16, "
                f"got {dtype}. Pass --kv-cache-dtype auto or use a bf16/fp16 model."
            )

        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device="cpu",          # force CPU; super will store self.device = "cpu"
            enable_memory_saver=False,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim

        self._create_buffers()
        self._finalize_allocation_log(size)

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _create_buffers(self) -> None:
        """Allocate CPU-pinned K and V buffers for each layer."""
        num_layers = self.end_layer - self.start_layer + 1
        k_shape = (self.size + 1, self.head_num, self.head_dim)
        v_shape = (self.size + 1, self.head_num, self.v_head_dim)

        total_bytes_k = (
            num_layers
            * (self.size + 1)
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        total_bytes_v = (
            num_layers
            * (self.size + 1)
            * self.head_num
            * self.v_head_dim
            * self.dtype.itemsize
        )
        total_gb = (total_bytes_k + total_bytes_v) / (1024**3)
        logger.info(
            f"[DirectKV] Allocating {num_layers} layers × "
            f"({self.size+1} tokens, {self.head_num} KV-heads, {self.head_dim} dim) "
            f"in CPU-pinned memory: {total_gb:.2f} GB"
        )

        self.k_buffer: list[torch.Tensor] = []
        self.v_buffer: list[torch.Tensor] = []

        for _ in range(num_layers):
            k = torch.zeros(k_shape, dtype=self.dtype, pin_memory=True)
            v = torch.zeros(v_shape, dtype=self.dtype, pin_memory=True)
            self.k_buffer.append(k)
            self.v_buffer.append(v)

    def _finalize_allocation_log(self, size: int) -> None:
        """Set mem_usage / token_stride and log the allocation (matches MHATokenToKVPool convention)."""
        num_layers = self.end_layer - self.start_layer + 1
        itemsize = self.dtype.itemsize
        k_bytes = num_layers * (size + 1) * self.head_num * self.head_dim * itemsize
        v_bytes = num_layers * (size + 1) * self.head_num * self.v_head_dim * itemsize
        self.mem_usage = (k_bytes + v_bytes) / (1024 ** 3)
        self.token_stride = self.head_num * self.head_dim

    # ------------------------------------------------------------------
    # KVCache interface
    # ------------------------------------------------------------------

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        num_layers = self.end_layer - self.start_layer + 1
        k_bytes = (
            num_layers
            * (self.size + 1)
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        v_bytes = (
            num_layers
            * (self.size + 1)
            * self.head_num
            * self.v_head_dim
            * self.dtype.itemsize
        )
        return k_bytes, v_bytes

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Return CPU-pinned K buffer for `layer_id`.  Shape: (size+1, H, D)."""
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """Return CPU-pinned V buffer for `layer_id`.  Shape: (size+1, H, Dv)."""
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,           # GPU int32/int64, shape (num_new_tokens,)
        cache_k: torch.Tensor,       # GPU fp16/bf16, shape (num_new_tokens, H, D)
        cache_v: torch.Tensor,       # GPU fp16/bf16, shape (num_new_tokens, H, Dv)
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ) -> None:
        """
        Scatter new K/V tokens from GPU into the CPU-pinned buffers.

        The copy is:
          1. D2H for the value tensors (GPU → pinned CPU)
          2. CPU-side scatter-write using integer indices

        This is a *synchronous* operation for the MVP (blocking D2H copy).
        An async variant using a pinned staging buffer is left as a TODO.
        """
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        li = layer_id - self.start_layer

        # Move indices to CPU (small copy, negligible overhead)
        loc_cpu = loc.cpu().long()

        # D2H copy (blocking): GPU tensors → CPU
        # For decode this is batch_size × head_num × head_dim × 2 bytes ≈ tiny
        k_cpu = cache_k.cpu()
        v_cpu = cache_v.cpu()

        # Scatter-write into pinned buffers at the correct token indices
        self.k_buffer[li][loc_cpu] = k_cpu
        self.v_buffer[li][loc_cpu] = v_cpu

    # ------------------------------------------------------------------
    # Compatibility stubs expected by token_to_kv_pool_allocator
    # ------------------------------------------------------------------

    def get_v_head_dim(self) -> int:
        return self.v_head_dim

    @property
    def same_kv_dim(self) -> bool:
        return self.head_dim == self.v_head_dim

    @property
    def row_dim(self) -> int:
        return self.head_num * self.head_dim

    # ------------------------------------------------------------------
    # Not-yet-supported methods (raise, don't silently break)
    # ------------------------------------------------------------------

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Copy KV blocks from src_loc to tgt_loc within the CPU pool."""
        tgt = tgt_loc.cpu().long()
        src = src_loc.cpu().long()
        for k_buf, v_buf in zip(self.k_buffer, self.v_buffer):
            k_buf[tgt] = k_buf[src].clone()
            v_buf[tgt] = v_buf[src].clone()

    def load_from_host_per_layer(
        self, host_pool, host_indices, device_indices, layer_id, io_backend
    ):
        li = layer_id - self.start_layer
        h_li = layer_id - host_pool.start_layer
        src_idx = host_indices.cpu().long() if not isinstance(host_indices, torch.Tensor) else host_indices.cpu().long()
        dst_idx = device_indices.cpu().long() if not isinstance(device_indices, torch.Tensor) else device_indices.cpu().long()
        self.k_buffer[li][dst_idx] = host_pool.k_buffer[h_li][src_idx]
        self.v_buffer[li][dst_idx] = host_pool.v_buffer[h_li][src_idx]

    def backup_to_host_all_layer(
        self, host_pool, host_indices, device_indices, io_backend
    ):
        h_idx = host_indices.cpu().long() if not isinstance(host_indices, torch.Tensor) else host_indices.cpu().long()
        d_idx = device_indices.cpu().long() if not isinstance(device_indices, torch.Tensor) else device_indices.cpu().long()
        for li in range(len(self.k_buffer)):
            layer_id = self.start_layer + li
            h_li = layer_id - host_pool.start_layer
            host_pool.k_buffer[h_li][h_idx] = self.k_buffer[li][d_idx]
            host_pool.v_buffer[h_li][h_idx] = self.v_buffer[li][d_idx]

    def get_cpu_copy(self, indices: torch.Tensor):
        raise NotImplementedError(
            "DirectKVTokenToKVPool: CPU is already the storage; no copy needed."
        )

    def load_cpu_copy(self, kv_cache_cpu, indices: torch.Tensor):
        raise NotImplementedError(
            "DirectKVTokenToKVPool: CPU is already the storage; no load needed."
        )
