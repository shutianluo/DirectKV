"""
DirectKV Attention Backend
============================
Fused K/V projection + Neox RoPE + CPU-KV attention via
sm_parallel_v2_cpu_kv_fwd_kernel (SM90, bf16, GH200 NVLink-C2C).

Key design points (see directkv_integrate.md):
  • Kernel does ALL projection (X→K, X→V) and K RoPE — no Python-side GEMM.
  • KV cache lives in CPU-pinned DirectKVRequestPool (contiguous per-request,
    no scatter/gather).
  • forward_decode  buckets requests by 64-aligned seqlen, one kernel call/bucket.
  • forward_extend  handles chunked prefill (--chunked-prefill-size multiple of 64).
  • CUDA graphs supported for DECODE mode via static max-shape TMA descriptors.

Constraints:
  head_dim == 128, bf16, SM90a (GH200).
  --disable-radix-cache required.
  --chunked-prefill-size must be a multiple of 64 (recommend 64 or 256).
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_TILE = 64   # kernel alignment requirement


def _pad64(n: int) -> int:
    return math.ceil(n / _TILE) * _TILE


# ---------------------------------------------------------------------------
# Metadata carried across init_forward_metadata → forward_decode/extend
# ---------------------------------------------------------------------------
@dataclass
class _FwdMeta:
    seq_lens_cpu: List[int]
    req_pool_indices_cpu: List[int]
    extend_seq_lens_cpu: Optional[List[int]]
    extend_prefix_lens_cpu: Optional[List[int]]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
class DirectKVBackend(AttentionBackend):

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()

        tp = get_attention_tp_size()
        cfg = model_runner.model_config

        self.num_q_heads  = cfg.num_attention_heads // tp
        self.num_kv_heads = cfg.get_num_kv_heads(tp)
        self.head_dim     = cfg.head_dim
        self.v_head_dim   = getattr(cfg, "v_head_dim", None) or cfg.head_dim
        self.hidden_size  = cfg.hidden_size
        self.device       = model_runner.device

        self._check_args(model_runner)

        # Load kernel extension once (JIT compile on first use)
        import torch.utils.cpp_extension  # noqa — must import before lazy load
        try:
            from sglang_ext.directkv_kernel import (
                directkv_fwd as _directkv_fwd,
                directkv_init_tma as _directkv_init_tma,
                directkv_fwd_graph as _directkv_fwd_graph,
            )
        except ImportError:
            from sglang.srt.layers.attention.directkv_kernel import (
                directkv_fwd as _directkv_fwd,
                directkv_init_tma as _directkv_init_tma,
                directkv_fwd_graph as _directkv_fwd_graph,
            )
        self._directkv_fwd       = _directkv_fwd
        self._directkv_init_tma  = _directkv_init_tma
        self._directkv_fwd_graph = _directkv_fwd_graph

        # Kernel supports MHA and GQA (NH_q must be a multiple of NH_kv).
        self._use_kernel: bool = (self.num_q_heads % self.num_kv_heads == 0)
        if not self._use_kernel:
            logger.warning(
                f"[DirectKV] NH_q={self.num_q_heads} is not a multiple of "
                f"NH_kv={self.num_kv_heads}. Falling back to SDPA."
            )

        # Weight / RoPE tables, filled by _setup_hooks()
        self._wk: Dict[int, torch.Tensor] = {}
        self._wv: Dict[int, torch.Tensor] = {}
        self._cos_sin: Optional[torch.Tensor] = None  # [max_pos, head_dim] fp32 GPU

        # Hidden states captured by pre-hook, keyed by layer_id
        self._hidden: Dict[int, torch.Tensor] = {}

        self._hooks_ready = False
        self._model_runner = model_runner
        self._fwd_meta: Optional[_FwdMeta] = None

        # Per-request contiguous KV pool for the kernel path (MHA only).
        # GQA/SDPA path uses forward_batch.token_to_kv_pool directly — no extra allocation.
        if self._use_kernel:
            try:
                from sglang_ext.directkv_request_pool import DirectKVRequestPool
            except ImportError:
                from sglang.srt.layers.attention.directkv_request_pool import DirectKVRequestPool
            self._pool_cls = DirectKVRequestPool
        else:
            self._pool_cls = None
        self._pool = None

        # Track active req_ids between steps for alloc/free (kernel path only)
        self._active_req_ids: set = set()

        # CUDA graph static tensors (populated in init_cuda_graph_state)
        # All dicts keyed by layer_id; allocated once and reused across replays.
        self._cg_max_bs:       int = 0
        self._cg_S_total_max:  int = 0            # max pool seqlen + _TILE
        self._cg_num_n_past:   Optional[torch.Tensor] = None  # [1] int32 GPU
        self._cg_num_n_blks:   Optional[torch.Tensor] = None  # [1] int32 GPU
        # Pool-row remapping [max_bs] int32 GPU: req_pool_indices[i] = pool row for batch i.
        # Updated before each replay (outside the graph), read by kernel during execution.
        self._cg_req_pool_indices: Optional[torch.Tensor] = None
        # Per-layer padded input/output tensors [max_bs, _TILE, ...]
        self._cg_X_pad:  Dict[int, torch.Tensor] = {}
        self._cg_Q_pad:  Dict[int, torch.Tensor] = {}
        self._cg_O_pad:  Dict[int, torch.Tensor] = {}
        # Pinned KV max tensors [n_pool, S_total_max, NH_kv, D] CPU — kept alive
        # so TMA descriptors don't reference freed memory.
        self._cg_K_max:  Dict[int, torch.Tensor] = {}
        self._cg_V_max:  Dict[int, torch.Tensor] = {}
        # Scatter/gather index for placing the current token in the 64-block
        # Shape [max_bs, 1, 1, 1] int64 — broadcast-compatible with [max_bs,_TILE,NH,D]
        self._cg_slot_idx:   Optional[torch.Tensor] = None  # [max_bs, 1, 1, 1] int64
        # start_layer (for pool layer indexing)
        self._start_layer: int = 0

    # ------------------------------------------------------------------
    # Argument validation
    # ------------------------------------------------------------------
    def _check_args(self, runner: "ModelRunner") -> None:
        sa = runner.server_args
        if not sa.disable_radix_cache:
            raise ValueError("directkv requires --disable-radix-cache")
        if self.head_dim != 128:
            raise ValueError(
                f"directkv requires head_dim=128, got {self.head_dim}. "
                "Use a compatible model (e.g. OPT-6.7B, Llama-3-8B with GQA mod)."
            )
        if runner.model_config.dtype not in (torch.bfloat16,):
            raise ValueError("directkv requires bfloat16 model weights.")

    # ------------------------------------------------------------------
    # Pool lazy init (kernel/MHA path only)
    # ------------------------------------------------------------------
    def _ensure_pool(self) -> None:
        if self._pool is not None or not self._use_kernel:
            return
        mr          = self._model_runner
        # Cap max_reqs: kernel pool only needs to cover the active decode batch.
        # 64 reqs × context_len × 32 layers × 2 × 32 heads × 128 dim ≈ 8 GB — manageable.
        # Use --max-running-requests to tune; default cap of 64 is safe for most GH200 runs.
        max_reqs    = min(mr.req_to_token_pool.size, 64)
        # Cap max_seqlen to the model's declared context length (avoid using the
        # full req_to_token table which may be padded to 2× context for safety).
        context_len = getattr(mr.model_config, "context_len", 2048)
        max_seqlen  = min(mr.req_to_token_pool.req_to_token.shape[1], context_len)
        num_layers  = mr.model_config.num_hidden_layers
        start_layer = getattr(mr, "start_layer", 0)
        self._start_layer = start_layer
        self._pool  = self._pool_cls(
            max_reqs    = max_reqs,
            max_seqlen  = max_seqlen,
            num_kv_heads= self.num_kv_heads,
            head_dim    = self.head_dim,
            num_layers  = num_layers,
            start_layer = start_layer,
        )
        logger.info(f"[DirectKV] Pool: {self._pool}")

    # ------------------------------------------------------------------
    # Hook setup and weight extraction
    # ------------------------------------------------------------------
    def _setup_hooks(self) -> None:
        model = self._model_runner.model
        self._start_layer = getattr(self._model_runner, "start_layer", 0)
        registered = 0
        for _name, module in model.named_modules():
            radix_attn = getattr(module, "attn", None)
            if radix_attn is None:
                continue
            layer_id = getattr(radix_attn, "layer_id", None)
            if layer_id is None:
                continue

            wk, wv = self._extract_weights(module, layer_id)
            if wk is None:
                continue
            self._wk[layer_id] = wk
            self._wv[layer_id] = wv

            if self._cos_sin is None:
                self._cos_sin = self._extract_cos_sin(module)

            lid = layer_id
            module.register_forward_pre_hook(
                lambda _m, args, kwargs, _lid=lid: self._capture_hs(_lid, args, kwargs),
                with_kwargs=True,
            )
            registered += 1

        logger.info(
            f"[DirectKV] Hooked {registered} layers; "
            f"RoPE={'yes' if self._cos_sin is not None else 'no (OPT)'}."
        )
        self._hooks_ready = True

    def _capture_hs(self, layer_id: int, args, kwargs=None) -> None:
        # Try kwargs first (Llama calls self_attn with keyword-only args)
        if kwargs and "hidden_states" in kwargs:
            self._hidden[layer_id] = kwargs["hidden_states"]
        elif args:
            self._hidden[layer_id] = args[0]

    def _extract_weights(
        self, module, layer_id: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        NH_kv = self.num_kv_heads
        NH_q  = self.num_q_heads
        D     = self.head_dim
        H     = self.hidden_size

        # LLaMA/Mistral: fused qkv_proj
        qkv = getattr(module, "qkv_proj", None)
        if qkv is not None and hasattr(qkv, "weight"):
            w   = qkv.weight                          # (NH_q*D + 2*NH_kv*D, H)
            q_s = NH_q  * D
            k_s = NH_kv * D
            wk  = w[q_s       : q_s + k_s].detach().contiguous().view(NH_kv, D, H)
            wv  = w[q_s + k_s : q_s + 2*k_s].detach().contiguous().view(NH_kv, D, H)
            return wk, wv

        # OPT/BERT: separate k_proj, v_proj
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)
        if k_proj is not None and v_proj is not None:
            if hasattr(k_proj, "weight") and hasattr(v_proj, "weight"):
                wk = k_proj.weight.detach().contiguous().view(NH_kv, D, H)
                wv = v_proj.weight.detach().contiguous().view(NH_kv, D, H)
                return wk, wv

        logger.warning(f"[DirectKV] Layer {layer_id}: cannot extract weights.")
        return None, None

    def _extract_cos_sin(self, module) -> Optional[torch.Tensor]:
        rotary = getattr(module, "rotary_emb", None)
        if rotary is None:
            return None
        D = self.head_dim

        # SGLang RotaryEmbedding stores a fused [max_pos, D] buffer: [:, :D//2]=cos, [:, D//2:]=sin
        cos_sin_cache = getattr(rotary, "cos_sin_cache", None)
        if cos_sin_cache is not None and cos_sin_cache.shape[-1] == D:
            return cos_sin_cache.float().to(self.device)

        for ca, sa in [("cos_cached", "sin_cached"), ("_cos_cached", "_sin_cached"),
                       ("cos", "sin")]:
            cos = getattr(rotary, ca, None)
            sin = getattr(rotary, sa, None)
            if cos is None or sin is None:
                continue
            while cos.dim() > 2:
                cos = cos.squeeze(0)
                sin = sin.squeeze(0)
            if cos.dim() == 2:
                hd = cos.shape[-1]
                if hd == D // 2:
                    return torch.cat([cos, sin], dim=-1).float().to(self.device)
                elif hd == D:
                    return torch.cat([cos[:, :D//2], sin[:, :D//2]], dim=-1).float().to(self.device)
        inv_freq = getattr(rotary, "inv_freq", None)
        if inv_freq is not None:
            max_pos = getattr(rotary, "max_position_embeddings", 8192)
            t  = torch.arange(max_pos, dtype=torch.float32, device=self.device)
            fq = torch.outer(t, inv_freq.float().to(self.device))
            return torch.cat([fq.cos(), fq.sin()], dim=-1)
        return None

    # ------------------------------------------------------------------
    # init_forward_metadata — called by model runner before each forward
    # ------------------------------------------------------------------
    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if forward_batch.forward_mode.is_idle():
            self._fwd_meta = None
            return

        if not self._hooks_ready:
            self._setup_hooks()

        seq_lens_cpu = forward_batch.seq_lens.cpu().tolist()
        req_ids_cpu  = forward_batch.req_pool_indices.cpu().tolist()

        if self._use_kernel:
            # Kernel path: manage DirectKVRequestPool row assignments.
            # Free finished rows FIRST so the pool has space before allocating new ones.
            self._ensure_pool()
            current_rids = set(req_ids_cpu)
            finished = self._active_req_ids - current_rids
            for rid in finished:
                self._pool.free(rid)
                self._active_req_ids.discard(rid)
            for rid in current_rids:
                if rid not in self._active_req_ids:
                    self._pool.alloc(rid)
                    self._active_req_ids.add(rid)

        self._fwd_meta = _FwdMeta(
            seq_lens_cpu         = seq_lens_cpu,
            req_pool_indices_cpu = req_ids_cpu,
            extend_seq_lens_cpu  = getattr(forward_batch, "extend_seq_lens_cpu", None),
            extend_prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None),
        )

    # ------------------------------------------------------------------
    # CUDA graph interface
    # ------------------------------------------------------------------
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        if not self._use_kernel:
            # GQA/SDPA path — no kernel graph support; no-op (SGLang won't graph it)
            self._cg_max_bs = max_bs
            return

        # Ensure weights are extracted before graph capture (may be called before any fwd pass)
        if not self._hooks_ready:
            self._setup_hooks()

        self._ensure_pool()

        self._cg_max_bs = max_bs

        NH_kv = self.num_kv_heads
        NH_q  = self.num_q_heads
        D     = self.head_dim
        H     = self.hidden_size

        # GPU int32 tensors for dynamic loop bounds (written before each replay)
        self._cg_num_n_past = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._cg_num_n_blks = torch.zeros(1, dtype=torch.int32, device=self.device)
        # Pool-row remapping [max_bs] — updated before each replay
        self._cg_req_pool_indices = torch.zeros(max_bs, dtype=torch.int32, device=self.device)

        # Scatter/gather slot index: [max_bs, 1, 1, 1] int64, broadcast over [max_bs, _TILE, NH, D]
        fill = self.get_cuda_graph_seq_len_fill_value()
        cap_slot = (fill - 1) % _TILE   # = _TILE - 1 = 63 during capture
        self._cg_slot_idx = torch.full(
            (max_bs, 1, 1, 1), cap_slot, dtype=torch.int64, device=self.device
        )

        # Per-layer padded tensors (static shapes, stable GPU pointers)
        max_seqlen_pad  = self._pool.max_seqlen_pad
        self._cg_S_total_max = max_seqlen_pad + _TILE
        for lid in list(self._wk.keys()):
            self._cg_X_pad[lid] = torch.zeros(
                max_bs, _TILE, H, dtype=torch.bfloat16, device=self.device
            )
            self._cg_Q_pad[lid] = torch.zeros(
                max_bs, _TILE, NH_q, D, dtype=torch.bfloat16, device=self.device
            )
            self._cg_O_pad[lid] = torch.zeros(
                max_bs, _TILE, NH_q, D, dtype=torch.bfloat16, device=self.device
            )

        # Pre-build TMA descriptors for all layers using pool max-shape views.
        # The K/V TMA descriptor uses n_pool rows (not max_bs) so the pool base is
        # the B dimension.  The kernel uses ptr_req_pool_indices to map batch pos → row.
        n_pool = self._pool.max_reqs
        wk_first_lid = min(self._wk.keys())
        for lid in list(self._wk.keys()):
            K_max, V_max = self._pool.get_cuda_graph_kv(lid, n_pool)
            # K_max: [n_pool, max_seqlen_pad, NH_kv, D]  CPU-pinned
            # We need K/V in shape [n_pool, S_total_max, NH_kv, D] and PINNED.
            # torch.cat() does NOT preserve pin_memory, so we explicitly allocate
            # a pinned buffer and copy.  The tensors are stored as instance attrs
            # (_cg_K_max / _cg_V_max) to prevent GC while TMA descriptors are live.
            S_total_max = self._cg_S_total_max
            if K_max.shape[1] < S_total_max:
                K_max_full = torch.empty(
                    n_pool, S_total_max, NH_kv, D,
                    dtype=K_max.dtype, pin_memory=True
                )
                K_max_full[:, :K_max.shape[1]].copy_(K_max)
                K_max_full[:, K_max.shape[1]:].zero_()
                V_max_full = torch.empty(
                    n_pool, S_total_max, NH_kv, D,
                    dtype=V_max.dtype, pin_memory=True
                )
                V_max_full[:, :V_max.shape[1]].copy_(V_max)
                V_max_full[:, V_max.shape[1]:].zero_()
            else:
                K_max_full = K_max[:, :S_total_max].contiguous()
                V_max_full = V_max[:, :S_total_max].contiguous()
            # Keep alive — TMA descriptors reference their raw memory
            self._cg_K_max[lid] = K_max_full
            self._cg_V_max[lid] = V_max_full

            wk = self._wk[lid]
            wv = self._wv[lid]
            self._directkv_init_tma(
                lid,
                self._cg_X_pad[lid],
                wk, wv,
                self._cg_Q_pad[lid],
                K_max_full,
                V_max_full,
            )
        logger.info(
            f"[DirectKV] CUDA graph state: max_bs={max_bs}, "
            f"S_total_max={self._cg_S_total_max}, layers={len(self._cg_X_pad)}."
        )

    def init_forward_metadata_capture_cuda_graph(
        self, bs, num_tokens, req_pool_indices, seq_lens,
        encoder_lens, forward_mode, spec_info,
    ) -> None:
        fill = self.get_cuda_graph_seq_len_fill_value()
        if self._use_kernel and self._cg_num_n_past is not None:
            S_past_cap  = fill - 1             # = 63
            n_past_cap  = S_past_cap // _TILE  # = 0 (since 63 < 64)
            n_blks_cap  = n_past_cap + 1       # = 1
            self._cg_num_n_past.fill_(n_past_cap)
            self._cg_num_n_blks.fill_(n_blks_cap)
            cap_slot = S_past_cap % _TILE      # = 63
            if self._cg_slot_idx is not None:
                self._cg_slot_idx.fill_(cap_slot)
        if self._use_kernel:
            rid_list = req_pool_indices.cpu().tolist()
            self._ensure_pool()
            for rid in rid_list:
                if rid not in self._active_req_ids:
                    self._pool.alloc(rid)
                    self._active_req_ids.add(rid)
            if self._cg_req_pool_indices is not None:
                row_list = [self._pool.row_of(rid) for rid in rid_list]
                self._cg_req_pool_indices[:bs].copy_(
                    torch.tensor(row_list, dtype=torch.int32))
        self._fwd_meta = _FwdMeta(
            seq_lens_cpu         = [fill] * bs,
            req_pool_indices_cpu = req_pool_indices.cpu().tolist(),
            extend_seq_lens_cpu  = None,
            extend_prefix_lens_cpu = None,
        )

    def init_forward_metadata_replay_cuda_graph(
        self, bs, req_pool_indices, seq_lens, seq_lens_sum,
        encoder_lens, forward_mode, spec_info, seq_lens_cpu=None,
    ) -> None:
        sl_cpu = (seq_lens_cpu.tolist() if seq_lens_cpu is not None
                  else seq_lens[:bs].cpu().tolist())

        if self._use_kernel and self._cg_num_n_past is not None:
            # All requests share the same padded seqlen for the kernel call.
            # Use the maximum S_past among active requests → single bucket.
            max_S_past    = max(sl - 1 for sl in sl_cpu) if sl_cpu else 0
            max_S_past_pad = (max_S_past // _TILE) * _TILE
            n_past = max_S_past_pad // _TILE
            n_blks = n_past + 1

            self._cg_num_n_past.fill_(n_past)
            self._cg_num_n_blks.fill_(n_blks)

            # common slot = (max_S_past % _TILE); same for all when all at same bucket
            common_slot = max_S_past % _TILE
            if self._cg_slot_idx is not None:
                self._cg_slot_idx.fill_(common_slot)

            rid_list = req_pool_indices[:bs].cpu().tolist()
            current_rids = set(rid_list)
            # Free rows for requests that are no longer active
            finished = self._active_req_ids - current_rids
            for rid in finished:
                self._pool.free(rid)
                self._active_req_ids.discard(rid)
            # Alloc pool rows for any new req ids
            for rid in current_rids:
                if rid not in self._active_req_ids:
                    self._pool.alloc(rid)
                    self._active_req_ids.add(rid)
            # Update pool-row remapping tensor for the kernel
            if self._cg_req_pool_indices is not None:
                row_list = [self._pool.row_of(rid) for rid in rid_list]
                self._cg_req_pool_indices[:bs].copy_(
                    torch.tensor(row_list, dtype=torch.int32))

        self._fwd_meta = _FwdMeta(
            seq_lens_cpu         = sl_cpu,
            req_pool_indices_cpu = req_pool_indices[:bs].cpu().tolist(),
            extend_seq_lens_cpu  = None,
            extend_prefix_lens_cpu = None,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return _TILE   # seqlen_past = _TILE-1=63 → n_past_cap=0, slot=63

    # ------------------------------------------------------------------
    # forward_decode
    # ------------------------------------------------------------------
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        lid   = layer.layer_id
        meta  = self._fwd_meta
        bs    = forward_batch.batch_size
        NH_q  = self.num_q_heads
        NH_kv = self.num_kv_heads
        D     = self.head_dim
        H     = self.hidden_size

        X_flat = self._hidden.pop(lid, None)
        wk     = self._wk.get(lid)
        wv     = self._wv.get(lid)

        if not self._use_kernel or X_flat is None or wk is None or meta is None:
            return self._sdpa_fallback(q, k, v, layer, forward_batch)

        # ------------------------------------------------------------------
        # CUDA graph fast path (uses pre-built TMA descriptors, no rebuild)
        # ------------------------------------------------------------------
        if (self._cg_num_n_past is not None
                and lid in self._cg_X_pad
                and forward_batch.forward_mode.is_decode()):
            return self._forward_decode_graph(
                lid, q, X_flat, layer, forward_batch, meta
            )

        # ------------------------------------------------------------------
        # Split requests: kernel-eligible (S_past % 64 == 0) vs SDPA fallback
        # The DirectKV kernel requires all past 64-blocks to be fully populated.
        # When S_past is not 64-aligned, positions [S_past:S_past_pad] in the
        # CPU pool are uninitialized zeros — use SDPA with exact masking instead.
        # ------------------------------------------------------------------
        sdpa_idxs   = []  # requests where S_past % 64 != 0
        kernel_idxs = []  # requests where S_past % 64 == 0

        for i in range(bs):
            S_past = meta.seq_lens_cpu[i] - 1
            if S_past % _TILE != 0:
                sdpa_idxs.append(i)
            else:
                kernel_idxs.append(i)

        output_slots: List[Tuple[int, torch.Tensor]] = []  # (i, O_i)

        # SDPA path: correct attention over exact [0:S_past+1] with DirectKV pool
        for i in sdpa_idxs:
            o_i = self._sdpa_decode_directkv(i, q, k, v, layer, meta)
            output_slots.append((i, o_i))

        # Kernel path: bucket 64-aligned requests → one directkv_fwd call per bucket
        if kernel_idxs:
            buckets: Dict[int, List[int]] = defaultdict(list)
            for i in kernel_idxs:
                S_past     = meta.seq_lens_cpu[i] - 1
                S_past_pad = _pad64(S_past) if S_past > 0 else 0
                buckets[S_past_pad].append(i)

            for S_past_pad, idxs in buckets.items():
                Bg          = len(idxs)
                S_new_pad   = _TILE
                S_total_pad = S_past_pad + S_new_pad

                X_pad = torch.zeros(Bg, S_new_pad, H,    dtype=torch.bfloat16, device=self.device)
                Q_pad = torch.zeros(Bg, S_new_pad, NH_q, D, dtype=torch.bfloat16, device=self.device)

                for gi, i in enumerate(idxs):
                    S_past = meta.seq_lens_cpu[i] - 1
                    slot   = S_past % _TILE
                    X_pad[gi, slot] = X_flat[i]
                    Q_pad[gi, slot] = q[i].view(NH_q, D)

                req_ids = [meta.req_pool_indices_cpu[i] for i in idxs]
                K_cpu_batch, V_cpu_batch = self._pool.get_batch_kv(req_ids, lid, S_total_pad)

                O = self._directkv_fwd(
                    X_pad, wk, wv, Q_pad,
                    K_cpu_batch, V_cpu_batch,
                    self._cos_sin,
                    float(layer.scaling),
                    True,
                    S_past_pad,
                    -1,
                )
                for gi, i in enumerate(idxs):
                    S_past = meta.seq_lens_cpu[i] - 1
                    slot   = S_past % _TILE
                    output_slots.append((i, O[gi, slot]))  # [NH_q, D]

        # Reassemble in original request order
        result = torch.empty(bs, NH_q * D, dtype=torch.bfloat16, device=self.device)
        for i, o_i in output_slots:
            result[i] = o_i.reshape(NH_q * D)
        return result

    # ------------------------------------------------------------------
    # forward_extend  (chunked prefill; --chunked-prefill-size % 64 == 0)
    # ------------------------------------------------------------------
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        lid   = layer.layer_id
        meta  = self._fwd_meta
        bs    = forward_batch.batch_size
        NH_q  = self.num_q_heads
        NH_kv = self.num_kv_heads
        D     = self.head_dim
        H     = self.hidden_size
        Dv    = self.v_head_dim

        X_all = self._hidden.pop(lid, None)
        wk    = self._wk.get(lid)
        wv    = self._wv.get(lid)

        if not self._use_kernel or X_all is None or wk is None or meta is None:
            return self._sdpa_extend_fallback(q, k, v, layer, forward_batch)

        ext_lens = meta.extend_seq_lens_cpu
        pre_lens = meta.extend_prefix_lens_cpu

        # Fallback if no chunked-prefill metadata (shouldn't happen but be safe)
        if ext_lens is None:
            return self._sdpa_extend_fallback(q, k, v, layer, forward_batch)

        outputs = []
        tok_off = 0
        for i in range(bs):
            sl_new = ext_lens[i]
            sl_pre = pre_lens[i] if pre_lens else 0

            if sl_new == 0 or sl_new % _TILE != 0 or sl_pre % _TILE != 0:
                # sl_new==0: empty request (warmup); else alignment violated — fall back to SDPA
                q_i = q[tok_off : tok_off + sl_new]
                k_i = k[tok_off : tok_off + sl_new]
                v_i = v[tok_off : tok_off + sl_new]
                o_i = self._sdpa_single(q_i, k_i, v_i, sl_new, sl_pre, layer,
                                        forward_batch, i, lid, save_kv_cache)
                outputs.append(o_i)
                tok_off += sl_new
                continue

            X_i = X_all[tok_off : tok_off + sl_new].unsqueeze(0)   # [1, sl_new, H]
            Q_i = q[tok_off : tok_off + sl_new]                     # [sl_new, NH_q*D]
            Q_i = Q_i.contiguous().view(1, sl_new, NH_q, D)         # [1, sl_new, NH_q, D]

            req_id  = meta.req_pool_indices_cpu[i]
            S_total = sl_pre + sl_new
            K_cpu, V_cpu = self._pool.get_kv(req_id, lid, S_total)  # [1, S_total, NH_kv, D]

            O_i = self._directkv_fwd(
                X_i, wk, wv, Q_i,
                K_cpu, V_cpu,
                self._cos_sin,
                float(layer.scaling),
                True,
                sl_pre,
                -1,
            )
            # O_i: [1, sl_new, NH_q, D]
            outputs.append(O_i.squeeze(0).reshape(sl_new, NH_q * D))
            tok_off += sl_new

        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # CUDA graph decode: single bucketed kernel call with pre-built descriptors
    # ------------------------------------------------------------------
    def _forward_decode_graph(
        self,
        lid: int,
        q: torch.Tensor,
        X_flat: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        meta: "_FwdMeta",
    ) -> torch.Tensor:
        """
        CUDA-graph-safe decode forward.

        All requests in the batch are treated as a single bucket at max S_past_pad
        (set in _cg_num_n_past by init_forward_metadata_replay_cuda_graph before
        replay). The slot index in _cg_slot_idx determines which position in the
        64-block receives the current token.

        CUDA ops in this method are captured in the graph on the first call (during
        graph capture) and replayed on subsequent calls. _cg_num_n_past,
        _cg_num_n_blks, and _cg_slot_idx are GPU tensors updated by the host
        BEFORE replay (outside the graph), so the kernel sees current-step values.
        """
        # bs is the actual capture/replay batch size (< max_bs for smaller graphs).
        # Slice all per-batch tensors to [bs, ...] so each graph sees its own shape.
        bs     = forward_batch.batch_size
        NH_q   = self.num_q_heads
        D      = self.head_dim
        H      = self.hidden_size

        # Pre-allocated [max_bs, _TILE, ...] — slice first bs rows for this graph
        X_pad  = self._cg_X_pad[lid][:bs]   # [bs, _TILE, H]
        Q_pad  = self._cg_Q_pad[lid][:bs]   # [bs, _TILE, NH_q, D]
        O_pad  = self._cg_O_pad[lid][:bs]   # [bs, _TILE, NH_q, D]
        si     = self._cg_slot_idx[:bs]      # [bs, 1, 1, 1] int64

        # Zero the padded buffers (cudaMemset — captured in graph)
        X_pad.zero_()
        Q_pad.zero_()

        # Scatter X_flat and Q into the slot position
        # X_flat: [bs, H] → scatter into [bs, _TILE, H]
        si_x = si.squeeze(-1).expand(bs, 1, H)          # [bs, 1, H]
        X_pad.scatter_(1, si_x, X_flat.view(bs, 1, H))

        # Q: [bs, NH_q*D] → scatter into [bs, _TILE, NH_q, D]
        si_q = si.expand(bs, 1, NH_q, D)                # [bs, 1, NH_q, D]
        Q_pad.scatter_(1, si_q, q[:bs].view(bs, 1, NH_q, D))

        # Kernel call (graph-safe: uses pre-built TMA descriptors).
        # seqlen_past_max is a static Python int (max pool seqlen) — OK to use
        # as a compile-time hint; the actual seqlen is read by the kernel from
        # _cg_num_n_past at device execution time.
        seqlen_past_static = self._cg_S_total_max - _TILE
        self._directkv_fwd_graph(
            lid,
            X_pad, Q_pad, O_pad,
            self._cg_num_n_past,
            self._cg_num_n_blks,
            self._cos_sin,
            float(layer.scaling),
            True,                       # is_causal
            self._cg_S_total_max,       # S_total_max (static, for TMA bounds)
            seqlen_past_static,         # seqlen_past_max hint (static)
            self._cg_req_pool_indices[:bs],  # pool-row remapping [bs] int32
        )

        # Gather output from slot position
        # O_pad: [bs, _TILE, NH_q, D] → gather → [bs, 1, NH_q, D]
        out = O_pad.gather(1, si_q).view(bs, NH_q * D)  # [bs, NH_q*D]
        return out

    # ------------------------------------------------------------------
    # SDPA fallbacks — manual per-request SDPA over CPU KV pool
    # ------------------------------------------------------------------
    def _sdpa_decode_directkv(
        self, idx: int,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        layer: "RadixAttention", meta: "_FwdMeta",
    ) -> torch.Tensor:
        """
        SDPA decode for a single request whose S_past is not 64-aligned.

        Reads past K/V from the DirectKV pool (positions [0:S_past]), appends
        the current token's K/V at position S_past, and runs scaled_dot_product_attention.
        This keeps all KV in the DirectKV pool so the kernel path can take over
        once S_past reaches the next 64 boundary.
        """
        import torch.nn.functional as F
        NH_q  = self.num_q_heads
        NH_kv = self.num_kv_heads
        D     = self.head_dim
        Dv    = self.v_head_dim
        lid   = layer.layer_id

        S_past = meta.seq_lens_cpu[idx] - 1        # tokens already in pool
        req_id = meta.req_pool_indices_cpu[idx]
        S_total = S_past + 1                        # past + current decode token

        # Write current K/V to DirectKV pool at position S_past
        K_c, V_c = self._pool.get_kv(req_id, lid, S_total)   # [1, S_total, NH_kv, D]
        k_cur = k[idx].view(NH_kv, D)
        v_cur = v[idx].view(NH_kv, Dv)
        K_c[0, S_past].copy_(k_cur.cpu())
        V_c[0, S_past].copy_(v_cur.cpu())

        # SDPA over exact [0:S_total] (no padding zeros in the attention window)
        k_4d = K_c[0, :S_total].permute(1, 0, 2).unsqueeze(0).to(self.device, non_blocking=True)
        v_4d = V_c[0, :S_total].permute(1, 0, 2).unsqueeze(0).to(self.device, non_blocking=True)
        q_4d = q[idx].view(1, NH_q, D).unsqueeze(0)          # [1, NH_q, 1, D]
        q_4d = q_4d.permute(0, 1, 2, 3)                      # already [1, NH_q, 1, D]
        q_4d = q[idx].view(1, 1, NH_q, D).permute(0, 2, 1, 3)

        if NH_q != NH_kv:
            k_4d = k_4d.repeat_interleave(NH_q // NH_kv, dim=1)
            v_4d = v_4d.repeat_interleave(NH_q // NH_kv, dim=1)

        o = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False,
                                           scale=layer.scaling)
        return o.squeeze().reshape(NH_q * Dv)  # [NH_q * D]

    def _sdpa_fallback(self, q, k, v, layer, forward_batch) -> torch.Tensor:
        """Decode SDPA (GQA or kernel-unavailable path)."""
        return self._sdpa_decode_manual(q, k, v, layer, forward_batch)

    def _sdpa_extend_fallback(self, q, k, v, layer, forward_batch) -> torch.Tensor:
        """Extend SDPA (GQA or misaligned prefill)."""
        return self._sdpa_extend_manual(q, k, v, layer, forward_batch)

    def _sdpa_decode_manual(self, q, k, v, layer, forward_batch) -> torch.Tensor:
        """Manual decode SDPA (kernel-path misalignment fallback using CPU KV pool)."""
        import torch.nn.functional as F
        meta = self._fwd_meta
        if meta is None:
            return torch.zeros(forward_batch.batch_size, self.num_q_heads * self.head_dim,
                               dtype=q.dtype, device=self.device)
        forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        bs, NH_q, NH_kv, D = forward_batch.batch_size, self.num_q_heads, self.num_kv_heads, self.head_dim
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_tok = forward_batch.req_to_token_pool.req_to_token
        outputs = []
        for i in range(bs):
            sl = forward_batch.seq_lens[i].item()
            slots = req_to_tok[forward_batch.req_pool_indices[i].item(), :sl].cpu()
            k_4d = k_buf[slots].permute(1, 0, 2).unsqueeze(0).to(self.device)
            v_4d = v_buf[slots].permute(1, 0, 2).unsqueeze(0).to(self.device)
            if NH_q != NH_kv:
                k_4d = k_4d.repeat_interleave(NH_q // NH_kv, dim=1)
                v_4d = v_4d.repeat_interleave(NH_q // NH_kv, dim=1)
            q_4d = q[i:i+1].view(1, 1, NH_q, D).permute(0, 2, 1, 3)
            o = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False, scale=layer.scaling)
            outputs.append(o.squeeze(0).squeeze(1).reshape(NH_q * D))
        return torch.stack(outputs, dim=0)

    def _sdpa_extend_manual(self, q, k, v, layer, forward_batch) -> torch.Tensor:
        """Manual extend SDPA (kernel-path misalignment fallback using CPU KV pool)."""
        import torch.nn.functional as F
        meta = self._fwd_meta
        if meta is None:
            return q.new_zeros(q.shape[0], self.num_q_heads * self.head_dim)
        forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        bs = forward_batch.batch_size
        NH_q, NH_kv, D, Dv = self.num_q_heads, self.num_kv_heads, self.head_dim, self.v_head_dim
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_tok = forward_batch.req_to_token_pool.req_to_token
        ext_lens = meta.extend_seq_lens_cpu or [meta.seq_lens_cpu[i] for i in range(bs)]
        pre_lens = meta.extend_prefix_lens_cpu or [0] * bs
        outputs, tok_off = [], 0
        q_flat = q.view(-1, NH_q, D)
        for i in range(bs):
            sl_new, sl_pre = ext_lens[i], pre_lens[i]
            sl_tot = sl_pre + sl_new
            slots = req_to_tok[forward_batch.req_pool_indices[i].item(), :sl_tot].cpu()
            k_4d = k_buf[slots].permute(1, 0, 2).unsqueeze(0).to(self.device)
            v_4d = v_buf[slots].permute(1, 0, 2).unsqueeze(0).to(self.device)
            if NH_q != NH_kv:
                k_4d = k_4d.repeat_interleave(NH_q // NH_kv, dim=1)
                v_4d = v_4d.repeat_interleave(NH_q // NH_kv, dim=1)
            q_4d = q_flat[tok_off:tok_off+sl_new].permute(1, 0, 2).unsqueeze(0)
            mask = torch.triu(torch.full((sl_new, sl_tot), float("-inf"), dtype=q.dtype, device=self.device), diagonal=sl_pre + 1).unsqueeze(0).unsqueeze(0)
            o = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, attn_mask=mask, scale=layer.scaling)
            outputs.append(o.squeeze(0).permute(1, 0, 2).reshape(sl_new, NH_q * Dv))
            tok_off += sl_new
        return torch.cat(outputs, dim=0)

    def _sdpa_single(self, q_i, k_i, v_i, sl_new, sl_pre, layer,
                     forward_batch, idx, lid, save_kv_cache) -> torch.Tensor:
        """SDPA for a single misaligned extend request in the kernel path (uses DirectKVRequestPool)."""
        import torch.nn.functional as F
        NH_q  = self.num_q_heads
        NH_kv = self.num_kv_heads
        D     = self.head_dim
        Dv    = self.v_head_dim
        sl_tot = sl_pre + sl_new
        req_id = self._fwd_meta.req_pool_indices_cpu[idx]

        k_new = k_i.view(sl_new, NH_kv, D)
        v_new = v_i.view(sl_new, NH_kv, Dv)
        K_c, V_c = self._pool.get_kv(req_id, lid, sl_tot)
        K_c[0, sl_pre:sl_tot].copy_(k_new.cpu())
        V_c[0, sl_pre:sl_tot].copy_(v_new.cpu())

        q_4d = q_i.view(1, sl_new, NH_q, D).permute(0, 2, 1, 3)
        k_4d = K_c.squeeze(0).permute(1, 0, 2).unsqueeze(0).to(self.device, non_blocking=True)
        v_4d = V_c.squeeze(0).permute(1, 0, 2).unsqueeze(0).to(self.device, non_blocking=True)
        if NH_q != NH_kv:
            g    = NH_q // NH_kv
            k_4d = k_4d.repeat_interleave(g, dim=1)
            v_4d = v_4d.repeat_interleave(g, dim=1)
        causal_mask = torch.triu(
            torch.full((sl_new, sl_tot), float("-inf"), dtype=q_i.dtype, device=self.device),
            diagonal=sl_pre + 1,
        ).unsqueeze(0).unsqueeze(0)
        o = F.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, attn_mask=causal_mask, scale=layer.scaling
        )
        return o.squeeze(0).permute(1,0,2).reshape(sl_new, NH_q * Dv)

    # init_cuda_graph_state, init_forward_metadata_capture_cuda_graph,
    # init_forward_metadata_replay_cuda_graph, and get_cuda_graph_seq_len_fill_value
    # are defined in the CUDA graph interface section above.
