"""
Useful buffers for model forward pass.
"""

import torch
from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.structs import SubBatch

class ModelForwardBuffers:
    """
    Useful buffers for model forward pass.
    """
    def __init__(
        self, 
        engine_config: EngineConfig,
        model_config: LlamaModelConfig
    ):
        iter_width = engine_config.max_tokens_in_batch
        hidden_size = model_config.hidden_size
        ws = model_config.world_size
        self.attn_out_buf = torch.zeros((iter_width, hidden_size // ws), dtype=torch.float16, device='cuda')
        self.residual_buf = torch.zeros((iter_width, hidden_size), dtype=torch.float16, device='cuda')
        self.cur_residual_buf = None

    def alloc_for_batches(self, batches: list[SubBatch]):
        """
        Allocate buffers for batches.
        """
        offs = 0
        for batch in batches:
            batch.attn_out_buf = self.attn_out_buf[offs: offs + batch.iter_width]
            batch.residual_buf = self.residual_buf[offs: offs + batch.iter_width]
            offs += batch.iter_width
        
        self.cur_residual_buf = self.residual_buf[:offs]
        self.cur_residual_buf.fill_(0.0)
