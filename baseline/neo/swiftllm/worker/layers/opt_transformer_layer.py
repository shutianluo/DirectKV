"""
OPT transformer layer: LayerNorm + MHA (no RoPE) + ReLU FFN.

Inherits the model-agnostic parts of LlamaTransformerLayer
(_attention, _transfer_qkv, _swap_out_blocks, set_swapper, forward)
and overrides _preproj / _postproj for OPT's architecture.
"""

import torch
import torch.nn.functional as F

from swiftllm_c import store_kvcache

from swiftllm.opt_model_config import OptModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.opt_weight import OptTransformerLayerWeight
from swiftllm.structs import SubBatch
from swiftllm.worker.kernels.linear import linear

from .transformer_layer import LlamaTransformerLayer


class OptTransformerLayer(LlamaTransformerLayer):
    """
    OPT transformer block.  Reuses the paged-attention logic from
    LlamaTransformerLayer; overrides projection helpers for OPT specifics:
      - LayerNorm (with bias) instead of RMSNorm
      - No RoPE
      - fc1/ReLU/fc2 instead of up·gate/SiLU/down
      - Bias on all linear projections
    """

    def __init__(
        self,
        model_config: OptModelConfig,
        engine_config: EngineConfig,
        weight: OptTransformerLayerWeight,
        next_layer_weight,
        cpu_communication_stream: torch.cuda.Stream,
        layer_id: int,
    ):
        # Call grandparent __init__ via object to skip LlamaTransformerLayer's
        # type-specific setup (there is none — __init__ just sets attributes).
        super().__init__(
            model_config, engine_config, weight, next_layer_weight,
            cpu_communication_stream, layer_id,
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _fused_add_layernorm_inplace(
        self,
        x: torch.Tensor,        # [N, H] fp16, modified in-place → LayerNorm output
        residual: torch.Tensor,  # [N, H] fp16, updated in-place → x_before_norm
        weight: torch.Tensor,    # [H]
        bias: torch.Tensor,      # [H]
        eps: float,
    ):
        x.add_(residual)
        residual.copy_(x)
        normed = F.layer_norm(
            x.float(), [x.shape[-1]], weight.float(), bias.float(), eps
        ).half()
        x.copy_(normed)

    # ------------------------------------------------------------------
    # Override _preproj: LayerNorm + QKV with bias, no RoPE
    # ------------------------------------------------------------------

    def _preproj(
        self,
        embeddings: torch.Tensor,
        batch: SubBatch,
        layer_off: int = 0,
    ):
        weight = self.weight if not layer_off else self.next_layer_weight

        self._maybe_allreduce(embeddings)

        self._fused_add_layernorm_inplace(
            embeddings, batch.residual_buf,
            weight.attn_norm_w, weight.attn_norm_b,
            self.model_config.layer_norm_eps,
        )

        # QKV projections with bias
        q = F.linear(embeddings, weight.q_proj, weight.q_bias)
        k = F.linear(embeddings, weight.k_proj, weight.k_bias)
        v = F.linear(embeddings, weight.v_proj, weight.v_bias)

        q = q.view(batch.iter_width, -1, self.model_config.head_dim)
        k = k.view(batch.iter_width, -1, self.model_config.head_dim)
        v = v.view(batch.iter_width, -1, self.model_config.head_dim)

        # No RoPE for OPT

        if batch.num_prefs > 0 and self.swapper is not None:
            gpu_layer = (self.layer_id + layer_off) % self.model_config.num_layers
            itm_layer = (self.model_config.num_layers
                         if self.engine_config.extra_layer_for_cprf
                         else gpu_layer)
            self._compute_wait_comm()
            store_kvcache(
                k, v,
                self.swapper.k_cache,
                self.swapper.v_cache,
                self.swapper.gpu_block_table,
                batch.prgd_seq_ids[:batch.num_prefs],
                batch.pref_st_locs_we,
                batch.prgd_seq_lens[:batch.num_prefs],
                itm_layer,
                gpu_layer,
                batch.num_cprfs,
                batch.max_pref_toks,
            )

        return q, k, v

    # ------------------------------------------------------------------
    # Override _postproj: LayerNorm + fc1/ReLU/fc2 with bias
    # ------------------------------------------------------------------

    def _postproj(self, batch: SubBatch) -> torch.Tensor:
        o = F.linear(batch.attn_out_buf, self.weight.o_proj, self.weight.o_bias)
        self._maybe_allreduce(o)

        self._fused_add_layernorm_inplace(
            o, batch.residual_buf,
            self.weight.ffn_norm_w, self.weight.ffn_norm_b,
            self.model_config.layer_norm_eps,
        )

        h = F.relu(F.linear(o, self.weight.fc1, self.weight.fc1_bias))
        del o
        embeddings = F.linear(h, self.weight.fc2, self.weight.fc2_bias)
        del h
        return embeddings
