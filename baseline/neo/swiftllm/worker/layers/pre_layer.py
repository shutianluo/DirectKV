"""
Pre-layer of the model.
"""
# pylint: disable=no-name-in-module

import torch

from swiftllm_c import embedding

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight


class LlamaPreLayer:
    """
    Pre-layer of the model.
    """
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
        seg_len = model_config.vocab_size // model_config.world_size
        self.token_offs = seg_len * model_config.rank
    
    def forward(
        self,
        input_ids: list[int]
    ) -> torch.Tensor:
        """
        Forward pass of the pre-layer.

        Each shard of the model is responsible for a segment of the vocabulary, and only sets the embeddings 
        for the tokens in its segment. For tokens outside of its segment, the embeddings are set to zeros.
        Then all the embeddings would be reduced across all the shards by the first transformer layer.
        """
        input_gpu = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
        embeddings = torch.zeros((len(input_ids), self.model_config.hidden_size), dtype=torch.float16, device='cuda')
        embedding(input_gpu, self.weights.wte, embeddings, self.token_offs)
        return embeddings
