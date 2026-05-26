"""
Pie-Style KV Cache Offloading for SGLang
==========================================
Implements performance-transparent KV cache swapping between GPU and CPU memory,
based on the Pie paper (UC Berkeley).

Core idea: LLM inference processes tokens **layer by layer**.  At any moment only
one layer's KV cache is actively used.  We exploit this predictable access pattern
to **prefetch upcoming layers from CPU → GPU** and **evict completed layers from
GPU → CPU** concurrently with computation, making swap latency invisible.

Architecture
------------
1. **MappingTable** — tracks per-layer KV cache residency (GPU / CPU / in-transit).
2. **SwapEngine**  — async DMA via dedicated CUDA streams + pinned CPU memory.
3. **FIFOSwapController** — FIFO policy deciding what to swap in / out at each
   layer computation step.  Never calls cudaEventSynchronize in the hot path
   (only cudaEventQuery).
4. **AdaptiveExpansionController** (Phase 2) — dynamically adjusts the number of
   CPU-resident layers to maximise throughput without stalling.
5. **PieKVPool** — drop-in subclass of ``MHATokenToKVPool`` that wraps a standard
   GPU pool + CPU shadow pool and routes ``get_kv_buffer / set_kv_buffer`` through
   the swap controller.

Usage
-----
Set ``--attention-backend pie`` (or monkey-patch via the startup hook).
The model-runner hook calls ``swap_controller.on_layer_compute_start(layer_id)``
before each transformer layer's attention, ensuring the correct KV buffers are on
GPU when needed.

Integration is non-invasive: only ``get_key_buffer`` / ``get_value_buffer`` are
overridden to redirect to the GPU shadow buffer for the current layer after the
swap controller has ensured residency.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)


# =====================================================================
# Component 1: Mapping Table
# =====================================================================

class LayerStatus(Enum):
    IN_GPU = "in_gpu"
    IN_CPU = "in_cpu"
    SWAPPING_IN = "swapping_in"
    SWAPPING_OUT = "swapping_out"


@dataclass
class LayerCacheEntry:
    """Tracks where one layer's KV cache physically resides."""
    layer_index: int
    status: LayerStatus = LayerStatus.IN_GPU
    last_access_step: int = 0  # global step counter of last compute access


class MappingTable:
    """
    Per-layer residency tracker.

    The table records which layers have their KV cache on GPU vs CPU (or
    in-transit).  The swap controller queries and updates this table.
    """

    def __init__(self, num_layers: int, num_gpu_layers: int):
        """
        Args:
            num_layers: total transformer layers.
            num_gpu_layers: how many layers start on GPU (the rest go to CPU).
        """
        self.num_layers = num_layers
        self.num_gpu_layers = num_gpu_layers
        self._entries: List[LayerCacheEntry] = []
        self._global_step = 0

        # Initialise: first num_gpu_layers on GPU, rest on CPU
        for i in range(num_layers):
            status = LayerStatus.IN_GPU if i < num_gpu_layers else LayerStatus.IN_CPU
            self._entries.append(LayerCacheEntry(layer_index=i, status=status))

    # -- queries --------------------------------------------------------

    def __getitem__(self, layer_id: int) -> LayerCacheEntry:
        return self._entries[layer_id]

    def is_on_gpu(self, layer_id: int) -> bool:
        return self._entries[layer_id].status == LayerStatus.IN_GPU

    def is_on_cpu(self, layer_id: int) -> bool:
        return self._entries[layer_id].status == LayerStatus.IN_CPU

    def gpu_layers(self) -> List[int]:
        return [e.layer_index for e in self._entries if e.status == LayerStatus.IN_GPU]

    def cpu_layers(self) -> List[int]:
        return [e.layer_index for e in self._entries if e.status == LayerStatus.IN_CPU]

    def num_cpu_resident(self) -> int:
        return sum(1 for e in self._entries if e.status in
                   (LayerStatus.IN_CPU, LayerStatus.SWAPPING_IN))

    # -- mutations ------------------------------------------------------

    def mark_accessed(self, layer_id: int):
        self._global_step += 1
        self._entries[layer_id].last_access_step = self._global_step

    def set_status(self, layer_id: int, status: LayerStatus):
        self._entries[layer_id].status = status

    @property
    def global_step(self) -> int:
        return self._global_step

    # -- FIFO helpers ---------------------------------------------------

    def get_coldest_gpu_layer(self, current_layer: int) -> Optional[int]:
        """
        Return the GPU-resident layer furthest from being needed next.

        In FIFO, the "coldest" layer is the one most recently computed
        (it won't be needed for another n-1 steps).  We use cyclic distance
        from *current_layer* as tie-breaker.

        Skip the current layer and layers that will be needed within a
        safe window (to avoid evicting something we're about to compute).
        """
        n = self.num_layers
        m_cpu = self.num_cpu_resident()
        if m_cpu == 0:
            return None  # no CPU layers → no FIFO rotation needed

        # Safe window: don't evict a layer if it's needed within m_cpu + 1
        # steps (we need those steps for the CPU layers to arrive).
        safe = m_cpu + 1

        def steps_until_needed(layer_idx):
            d = (layer_idx - current_layer) % n
            return d if d > 0 else n

        candidates = [
            e for e in self._entries
            if e.status == LayerStatus.IN_GPU
            and e.layer_index != current_layer
            and steps_until_needed(e.layer_index) > safe
        ]
        if not candidates:
            return None
        # Coldest = furthest from being needed
        coldest = max(candidates, key=lambda e: steps_until_needed(e.layer_index))
        return coldest.layer_index

    def get_hottest_cpu_layer(self, current_layer: int) -> Optional[int]:
        """
        Return the CPU-resident layer closest to being needed next.

        Skip the current layer (already being handled).
        """
        cpu = [
            e for e in self._entries
            if e.status == LayerStatus.IN_CPU and e.layer_index != current_layer
        ]
        if not cpu:
            return None
        n = self.num_layers
        def dist(layer_idx):
            d = (layer_idx - current_layer) % n
            return d if d > 0 else n  # current layer → max distance
        hottest = min(cpu, key=lambda e: dist(e.layer_index))
        return hottest.layer_index


# =====================================================================
# Component 2: Swap Engine
# =====================================================================

class SwapEngine:
    """
    Async GPU↔CPU data transfer engine using dedicated CUDA streams.

    All CPU tensors MUST be pinned.  Transfers are issued with
    ``non_blocking=True`` on their respective streams.  Completion is
    checked via ``cudaEventQuery`` (non-blocking), never
    ``cudaEventSynchronize``, in the hot path.
    """

    def __init__(self, device: torch.device):
        self.device = device
        # Dedicated streams — run concurrently with the default compute stream.
        self.stream_in = torch.cuda.Stream(device=device)   # CPU → GPU
        self.stream_out = torch.cuda.Stream(device=device)  # GPU → CPU
        # Lightweight events for non-blocking completion checks.
        self.event_in = torch.cuda.Event(enable_timing=False)
        self.event_out = torch.cuda.Event(enable_timing=False)
        # Track what is currently in flight.
        self._in_flight_in: Optional[int] = None   # layer_id being swapped in
        self._in_flight_out: Optional[int] = None  # layer_id being swapped out

    # -- swap in (CPU → GPU) -------------------------------------------

    def swap_in_async(
        self,
        layer_id: int,
        cpu_k: torch.Tensor,
        cpu_v: torch.Tensor,
        gpu_k: torch.Tensor,
        gpu_v: torch.Tensor,
    ):
        """Start an async CPU→GPU copy for *layer_id*."""
        # Ensure the swap stream sees any prior writes on the compute stream.
        self.stream_in.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream_in):
            gpu_k.copy_(cpu_k, non_blocking=True)
            gpu_v.copy_(cpu_v, non_blocking=True)
        self.event_in.record(self.stream_in)
        self._in_flight_in = layer_id

    def is_swap_in_done(self) -> bool:
        if self._in_flight_in is None:
            return True
        return self.event_in.query()

    def finish_swap_in(self) -> Optional[int]:
        """If the swap-in completed, return the layer_id and clear state."""
        if self._in_flight_in is None:
            return None
        if self.event_in.query():
            lid = self._in_flight_in
            self._in_flight_in = None
            return lid
        return None

    def wait_swap_in(self):
        """Blocking wait — called only when compute MUST use a layer that
        hasn't arrived yet.  This is what adaptive expansion tries to avoid."""
        if self._in_flight_in is not None:
            self.event_in.synchronize()
            self._in_flight_in = None

    # -- swap out (GPU → CPU) ------------------------------------------

    def swap_out_async(
        self,
        layer_id: int,
        gpu_k: torch.Tensor,
        gpu_v: torch.Tensor,
        cpu_k: torch.Tensor,
        cpu_v: torch.Tensor,
    ):
        """Start an async GPU→CPU copy for *layer_id*."""
        # Ensure the swap stream sees any prior writes on the compute stream
        # (the GPU tensor must be fully computed before we read it).
        self.stream_out.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream_out):
            cpu_k.copy_(gpu_k, non_blocking=True)
            cpu_v.copy_(gpu_v, non_blocking=True)
        self.event_out.record(self.stream_out)
        self._in_flight_out = layer_id

    def is_swap_out_done(self) -> bool:
        if self._in_flight_out is None:
            return True
        return self.event_out.query()

    def finish_swap_out(self) -> Optional[int]:
        """If the swap-out completed, return the layer_id and clear state."""
        if self._in_flight_out is None:
            return None
        if self.event_out.query():
            lid = self._in_flight_out
            self._in_flight_out = None
            return lid
        return None

    @property
    def in_flight_in_layer(self) -> Optional[int]:
        return self._in_flight_in

    @property
    def in_flight_out_layer(self) -> Optional[int]:
        return self._in_flight_out


# =====================================================================
# Component 3: FIFO Swap Controller
# =====================================================================

class FIFOSwapController:
    """
    Decides what to swap in / out at the start of each layer's computation.

    Algorithm (runs at ``on_layer_compute_start``):
    1. Check if the in-flight swap-out completed → update mapping table.
    2. Check if the in-flight swap-in completed → update mapping table.
       Possibly start a new swap-out (coldest GPU layer).
       Possibly start a new swap-in (hottest CPU layer).
    3. Ensure the current layer is on GPU (block if necessary — recorded as
       a *stall* for the adaptive controller).

    The controller never calls ``cudaEventSynchronize`` in steps 1–2;
    it only calls ``cudaEventQuery``.  Step 3 may block if a layer is still
    in transit — this is the scenario adaptive expansion seeks to eliminate.
    """

    def __init__(
        self,
        mapping: MappingTable,
        engine: SwapEngine,
        gpu_k_buffers: List[torch.Tensor],
        gpu_v_buffers: List[torch.Tensor],
        cpu_k_buffers: List[torch.Tensor],
        cpu_v_buffers: List[torch.Tensor],
        start_layer: int = 0,
    ):
        self.mapping = mapping
        self.engine = engine
        self.gpu_k = gpu_k_buffers   # per-layer GPU KV buffers
        self.gpu_v = gpu_v_buffers
        self.cpu_k = cpu_k_buffers   # per-layer CPU-pinned KV buffers
        self.cpu_v = cpu_v_buffers
        self.start_layer = start_layer
        self.stall_count = 0
        self.idle_swap_in = 0
        self.idle_swap_out = 0

    def _buf_idx(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    # -- main entry point (called before each layer) -------------------

    def on_layer_compute_start(self, current_layer: int) -> bool:
        """
        Called before layer *current_layer* begins attention computation.
        Returns True if compute had to stall waiting for a swap-in.
        """
        had_stall = False

        # 1. Finalise completed swap-out
        done_out = self.engine.finish_swap_out()
        if done_out is not None:
            self.mapping.set_status(done_out, LayerStatus.IN_CPU)

        # 2. Finalise completed swap-in
        done_in = self.engine.finish_swap_in()
        if done_in is not None:
            self.mapping.set_status(done_in, LayerStatus.IN_GPU)

        # 3. Initiate new swap-out if stream is idle
        if self.engine.is_swap_out_done():
            coldest = self.mapping.get_coldest_gpu_layer(current_layer)
            if coldest is not None and self.mapping.num_cpu_resident() > 0:
                # Only evict if there are CPU-resident layers (FIFO: coldest GPU
                # should be colder than all CPU layers)
                idx = self._buf_idx(coldest)
                self.engine.swap_out_async(
                    coldest,
                    self.gpu_k[idx], self.gpu_v[idx],
                    self.cpu_k[idx], self.cpu_v[idx],
                )
                self.mapping.set_status(coldest, LayerStatus.SWAPPING_OUT)
            else:
                self.idle_swap_out += 1

        # 4. Initiate new swap-in if stream is idle
        if self.engine.is_swap_in_done():
            hottest = self.mapping.get_hottest_cpu_layer(current_layer)
            if hottest is not None:
                idx = self._buf_idx(hottest)
                self.engine.swap_in_async(
                    hottest,
                    self.cpu_k[idx], self.cpu_v[idx],
                    self.gpu_k[idx], self.gpu_v[idx],
                )
                self.mapping.set_status(hottest, LayerStatus.SWAPPING_IN)
            else:
                self.idle_swap_in += 1

        # 5. Ensure current layer is on GPU
        entry = self.mapping[current_layer]
        if entry.status == LayerStatus.SWAPPING_IN:
            # Must wait for the in-flight swap-in to complete
            self.engine.wait_swap_in()
            self.mapping.set_status(current_layer, LayerStatus.IN_GPU)
            had_stall = True
            self.stall_count += 1
        elif entry.status == LayerStatus.IN_CPU:
            # Needs immediate (blocking) swap-in — worst case
            idx = self._buf_idx(current_layer)
            self.engine.swap_in_async(
                current_layer,
                self.cpu_k[idx], self.cpu_v[idx],
                self.gpu_k[idx], self.gpu_v[idx],
            )
            self.engine.wait_swap_in()
            self.mapping.set_status(current_layer, LayerStatus.IN_GPU)
            had_stall = True
            self.stall_count += 1
        elif entry.status == LayerStatus.SWAPPING_OUT:
            # Edge case: we're about to compute on a layer being evicted.
            # Wait for eviction to finish, then swap it back in.
            self.engine.event_out.synchronize()
            self.mapping.set_status(current_layer, LayerStatus.IN_CPU)
            self.engine._in_flight_out = None
            # Now swap it back in (blocking)
            idx = self._buf_idx(current_layer)
            self.engine.swap_in_async(
                current_layer,
                self.cpu_k[idx], self.cpu_v[idx],
                self.gpu_k[idx], self.gpu_v[idx],
            )
            self.engine.wait_swap_in()
            self.mapping.set_status(current_layer, LayerStatus.IN_GPU)
            had_stall = True
            self.stall_count += 1

        # Mark this layer as accessed (for FIFO ordering)
        self.mapping.mark_accessed(current_layer)
        return had_stall

    def reset_counters(self):
        stalls = self.stall_count
        idle_in = self.idle_swap_in
        idle_out = self.idle_swap_out
        self.stall_count = 0
        self.idle_swap_in = 0
        self.idle_swap_out = 0
        return stalls, idle_in, idle_out


# =====================================================================
# Component 4: Adaptive Expansion Controller (Phase 2)
# =====================================================================

class AdaptiveExpansionController:
    """
    Dynamically adjusts *m* (number of CPU-resident layers) at runtime.

    State machine:
    - Start at m = initial_m (from the fixed expansion ratio).
    - Every ``check_interval`` forward passes through all layers, examine
      stall_count and idle bandwidth.
    - Increase m if idle bandwidth is high and throughput improves.
    - Decrease m if stalls are frequent.
    - Keep at least 2 layers on GPU at all times.
    """

    def __init__(
        self,
        num_layers: int,
        initial_m: int,
        check_interval: int = 5,
        stall_threshold: float = 0.1,
        idle_threshold: float = 0.5,
    ):
        self.num_layers = num_layers
        self.m = initial_m
        self.check_interval = check_interval
        self.stall_threshold = stall_threshold
        self.idle_threshold = idle_threshold
        self.pass_count = 0
        self.throughput_history: List[float] = []
        self._pending_realloc: Optional[int] = None

    def on_forward_pass_complete(
        self,
        stall_count: int,
        idle_swap_in: int,
        idle_swap_out: int,
        tokens_processed: int,
        elapsed_s: float,
    ) -> Optional[int]:
        """
        Called after one full forward pass through all layers.
        Returns the new *m* if it changed, else None.
        """
        self.pass_count += 1
        throughput = tokens_processed / elapsed_s if elapsed_s > 0 else 0.0
        self.throughput_history.append(throughput)

        if self.pass_count % self.check_interval != 0:
            return None

        total_events = self.num_layers * self.check_interval
        stall_ratio = stall_count / total_events if total_events > 0 else 0.0
        idle_ratio = (idle_swap_in + idle_swap_out) / (2 * total_events) if total_events > 0 else 0.0

        old_m = self.m

        if stall_ratio > self.stall_threshold:
            # Too many stalls — reduce offloaded layers
            self.m = max(0, self.m - 1)
        elif idle_ratio > self.idle_threshold:
            # Idle bandwidth — can offload more
            if self._throughput_improving():
                self.m = min(self.num_layers - 2, self.m + 1)
            # else: compute-bound, don't expand

        if self.m != old_m:
            logger.info(
                f"[Pie] Adaptive expansion: m {old_m} → {self.m} "
                f"(stall_ratio={stall_ratio:.2f}, idle_ratio={idle_ratio:.2f})"
            )
            return self.m
        return None

    def _throughput_improving(self) -> bool:
        if len(self.throughput_history) < 2 * self.check_interval:
            return True  # not enough data, assume improvement
        recent = self.throughput_history[-self.check_interval:]
        prev = self.throughput_history[-2 * self.check_interval:-self.check_interval]
        return sum(recent) >= sum(prev) * 0.98  # at least 98% of previous


# =====================================================================
# Component 5: PieKVPool — KV pool with layer-level GPU/CPU swapping
# =====================================================================

class PieKVPool:
    """
    Wraps a standard ``MHATokenToKVPool`` (GPU) with a CPU-pinned shadow pool
    and the Pie swap controller.

    The GPU pool keeps KV buffers for *all* layers, but only ``n - m`` of them
    hold live data at any time.  The remaining ``m`` layers' data lives in the
    CPU shadow pool and is swapped in before it's needed.

    ``get_key_buffer`` / ``get_value_buffer`` always return the GPU buffer — the
    swap controller guarantees it contains the correct data by the time attention
    computes.
    """

    def __init__(
        self,
        gpu_pool,   # MHATokenToKVPool
        expansion_ratio: float = 1.3,
        device: Optional[torch.device] = None,
    ):
        self.gpu_pool = gpu_pool
        self.num_layers = gpu_pool.layer_num
        self.start_layer = gpu_pool.start_layer
        self.device = device or torch.device(gpu_pool.device)

        # Compute m (number of layers offloaded to CPU)
        # expansion_ratio = n / (n - m)  →  m = n * (1 - 1/expansion_ratio)
        self.m = max(0, round(self.num_layers * (1.0 - 1.0 / expansion_ratio)))
        self.m = min(self.m, self.num_layers - 2)  # keep ≥2 on GPU
        num_gpu = self.num_layers - self.m

        logger.info(
            f"[Pie] layers={self.num_layers}, m={self.m}, "
            f"gpu_layers={num_gpu}, expansion={self.num_layers / num_gpu:.2f}×"
        )

        # CPU shadow pool: pinned memory, same shape as GPU buffers
        self.cpu_k_buffers: List[torch.Tensor] = []
        self.cpu_v_buffers: List[torch.Tensor] = []
        for i in range(self.num_layers):
            k_gpu = gpu_pool._get_key_buffer(i + self.start_layer)
            v_gpu = gpu_pool._get_value_buffer(i + self.start_layer)
            self.cpu_k_buffers.append(
                torch.zeros_like(k_gpu, device="cpu", pin_memory=True)
            )
            self.cpu_v_buffers.append(
                torch.zeros_like(v_gpu, device="cpu", pin_memory=True)
            )

        # References to GPU buffers
        self.gpu_k_buffers = gpu_pool.k_buffer
        self.gpu_v_buffers = gpu_pool.v_buffer

        # Mapping table
        self.mapping = MappingTable(self.num_layers, num_gpu)

        # Swap engine
        self.engine = SwapEngine(self.device)

        # FIFO swap controller
        self.controller = FIFOSwapController(
            mapping=self.mapping,
            engine=self.engine,
            gpu_k_buffers=self.gpu_k_buffers,
            gpu_v_buffers=self.gpu_v_buffers,
            cpu_k_buffers=self.cpu_k_buffers,
            cpu_v_buffers=self.cpu_v_buffers,
            start_layer=self.start_layer,
        )

        # Adaptive expansion (Phase 2)
        self.adaptive = AdaptiveExpansionController(
            num_layers=self.num_layers,
            initial_m=self.m,
        )

        # Initialise CPU buffers for offloaded layers:
        # Copy initial GPU buffer contents → CPU for layers starting on CPU.
        for layer_id in self.mapping.cpu_layers():
            idx = layer_id  # self.start_layer already handled in mapping
            self.cpu_k_buffers[idx].copy_(self.gpu_k_buffers[idx])
            self.cpu_v_buffers[idx].copy_(self.gpu_v_buffers[idx])

        # Stats
        self._pass_start_time: Optional[float] = None
        self._tokens_this_pass = 0

    # -- Public API (called from attention backend / hook) ---------------

    def on_layer_compute_start(self, layer_id: int) -> bool:
        """Call before layer *layer_id* begins attention.  Returns had_stall."""
        local_id = layer_id - self.start_layer
        if local_id == 0:
            self._pass_start_time = time.monotonic()
        return self.controller.on_layer_compute_start(local_id)

    def on_forward_pass_complete(self, tokens_processed: int = 0):
        """Call after one full forward pass.  Drives adaptive expansion."""
        elapsed = (time.monotonic() - self._pass_start_time
                   if self._pass_start_time else 0.0)
        stalls, idle_in, idle_out = self.controller.reset_counters()
        new_m = self.adaptive.on_forward_pass_complete(
            stall_count=stalls,
            idle_swap_in=idle_in,
            idle_swap_out=idle_out,
            tokens_processed=tokens_processed,
            elapsed_s=elapsed,
        )
        if new_m is not None and new_m != self.m:
            self._realloc(new_m)

    def _realloc(self, new_m: int):
        """Change the number of CPU-resident layers."""
        old_m = self.m
        self.m = new_m
        # Rebuild mapping
        num_gpu = self.num_layers - self.m
        self.mapping = MappingTable(self.num_layers, num_gpu)
        self.controller.mapping = self.mapping
        # Copy GPU→CPU for newly offloaded layers
        for layer_id in self.mapping.cpu_layers():
            self.cpu_k_buffers[layer_id].copy_(self.gpu_k_buffers[layer_id])
            self.cpu_v_buffers[layer_id].copy_(self.gpu_v_buffers[layer_id])
        logger.info(f"[Pie] Reallocated: m {old_m} → {new_m}")


# =====================================================================
# Hook into SGLang's model runner
# =====================================================================

_pie_pool: Optional[PieKVPool] = None


def get_pie_pool() -> Optional[PieKVPool]:
    return _pie_pool


def set_pie_pool(pool: PieKVPool):
    global _pie_pool
    _pie_pool = pool


def install_pie_hook():
    """
    Monkey-patch ``MHATokenToKVPool.get_key_buffer`` and
    ``get_value_buffer`` so the swap controller is called before each
    layer's KV cache is read.

    This is the main integration point — no other SGLang files need to be
    modified.
    """
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    orig_get_key = MHATokenToKVPool.get_key_buffer
    orig_get_val = MHATokenToKVPool.get_value_buffer

    def hooked_get_key(self, layer_id: int):
        pool = get_pie_pool()
        if pool is not None and pool.gpu_pool is self:
            pool.on_layer_compute_start(layer_id)
        return orig_get_key(self, layer_id)

    def hooked_get_val(self, layer_id: int):
        pool = get_pie_pool()
        if pool is not None and pool.gpu_pool is self:
            # Already ensured by hooked_get_key (called first in every backend)
            pass
        return orig_get_val(self, layer_id)

    MHATokenToKVPool.get_key_buffer = hooked_get_key
    MHATokenToKVPool.get_value_buffer = hooked_get_val
    logger.info("[Pie] Installed get_kv_buffer hooks on MHATokenToKVPool.")


def create_pie_pool(gpu_pool, expansion_ratio: float = 1.3):
    """Create a PieKVPool wrapping *gpu_pool* and install the hook."""
    pool = PieKVPool(gpu_pool, expansion_ratio=expansion_ratio)
    set_pie_pool(pool)
    install_pie_hook()
    return pool
