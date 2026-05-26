"""
Post layer of the model.
"""

import torch
import torch.distributed as dist

# pylint: disable=no-name-in-module
from swiftllm_c import fused_add_rmsnorm_inplace

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight
# from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace
from swiftllm.worker.kernels.linear import linear
from swiftllm.structs import SubBatch

class LlamaPostLayer:
    """
    Post layer of the model.
    """
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    
    def forward(
        self,
        batches: list[SubBatch],
        input_embeds: torch.Tensor,	# [num_total_tokens, hidden_size]
        residual_buf: torch.Tensor 	# [num_total_tokens, hidden_size]
    ) -> list[int]:
        """
        Forward pass of the post layer.
        """
        offs = 0
        for batch in batches:
            last_token_indices = torch.cat(
                (
                    last_token_indices,
                    batch.last_token_indices + offs
                ), dim=0
            ) if offs else batch.last_token_indices
            offs += batch.iter_width

        input_embeds = input_embeds[last_token_indices, :]
        residual_buf = residual_buf[last_token_indices, :]

        if self.model_config.world_size > 1:
            dist.all_reduce(input_embeds)
        
        fused_add_rmsnorm_inplace(
            input_embeds,
            residual_buf,
            self.weights.final_norm,
            self.model_config.rms_norm_eps
        )

        logits = linear(input_embeds, self.weights.lm_head)    # [batch_size, vocab_size]
        if self.model_config.world_size > 1:
            gather_list = [torch.zeros_like(logits) for _ in range(self.model_config.world_size)] \
                if self.model_config.rank == 0 else None
            dist.gather(logits, gather_list) # only rank 0 will have the final logits
            if self.model_config.rank == 0:
                logits = torch.cat(gather_list, dim=1)
        return torch.argmax(logits, dim=1).tolist() if self.model_config.rank == 0 else []
    