# Inference-only OPT model for SGLang.
# Architecture follows facebook/opt-* (MHA, learned absolute pos embeddings,
# pre-norm when do_layer_norm_before=True, ReLU FFN).
# Weight keys match the HuggingFace OPTForCausalLM state-dict.
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import OPTConfig

from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

# OPT position embedding offset: HF adds (pad_token_id + 1) = 2.
_OPT_POS_OFFSET = 2


class OPTAttention(nn.Module):

    def __init__(
        self,
        layer_id: int,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.hidden_size  = config.hidden_size
        self.total_heads  = config.num_attention_heads
        self.num_heads    = self.total_heads // tp
        self.head_dim     = self.hidden_size // self.total_heads
        self.scale        = self.head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(
            self.hidden_size, self.hidden_size, bias=True,
            quant_config=quant_config, prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, self.hidden_size, bias=True,
            quant_config=quant_config, prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, self.hidden_size, bias=True,
            quant_config=quant_config, prefix=add_prefix("v_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            self.hidden_size, self.hidden_size, bias=True,
            quant_config=quant_config, prefix=add_prefix("out_proj", prefix),
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scale,
            num_kv_heads=self.num_heads,   # MHA: kv_heads == q_heads per shard
            layer_id=layer_id,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):

    def __init__(
        self,
        layer_id: int,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn = OPTAttention(
            layer_id, config, quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size, config.ffn_dim, bias=True,
            quant_config=quant_config, prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.ffn_dim, config.hidden_size, bias=True,
            quant_config=quant_config, prefix=add_prefix("fc2", prefix),
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Self-attention (with pre-norm or post-norm)
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, forward_batch)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # FFN
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.word_embed_proj_dim,
        )
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings + _OPT_POS_OFFSET,
            config.hidden_size,
        )
        # Project from word_embed_proj_dim → hidden_size when they differ (OPT-125M)
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in  = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_in  = None
            self.project_out = None

        self.layers = nn.ModuleList([
            OPTDecoderLayer(
                layer_id=i,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix(f"layers.{i}", prefix),
            )
            for i in range(config.num_hidden_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self._do_final_norm = not config.do_layer_norm_before

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Token + position embeddings
        # SGLang passes 0-based positions; OPT embed_positions expects +2 offset.
        inputs_embeds = self.embed_tokens(input_ids)
        pos_embeds    = self.embed_positions(positions + _OPT_POS_OFFSET)
        hidden_states = inputs_embeds + pos_embeds

        if self.project_in is not None:
            hidden_states = self.project_in(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, forward_batch)

        hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


class OPTForCausalLM(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config       = config
        self.quant_config = quant_config
        self.model = OPTDecoder(
            config, quant_config, prefix=add_prefix("model.decoder", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.word_embed_proj_dim,
            quant_config=quant_config, prefix=add_prefix("lm_head", prefix),
        )
        # OPT ties lm_head to embed_tokens
        self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            # Skip tied lm_head duplicate (loaded via embed_tokens)
            if name == "lm_head.weight":
                continue
            # HF safetensors use "model.decoder.X"; PyTorch named_parameters use "model.X"
            if name.startswith("model.decoder."):
                name = "model." + name[len("model.decoder."):]
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = OPTForCausalLM
