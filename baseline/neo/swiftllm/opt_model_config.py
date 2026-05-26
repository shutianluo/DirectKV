import os
import json
import torch


class OptModelConfig:
    """
    Configuration for an OPT model (facebook/opt-*).
    """

    def __init__(self, cfg: dict):
        assert cfg["model_type"] == "opt", f"Expected model_type=opt, got {cfg['model_type']}"
        self.num_layers = cfg["num_hidden_layers"]
        self.num_q_heads = cfg["num_attention_heads"]
        self.num_kv_heads = self.num_q_heads          # OPT has no GQA
        self.hidden_size = cfg["hidden_size"]
        self.head_dim = self.hidden_size // self.num_q_heads
        self.vocab_size = cfg["vocab_size"]
        self.max_position_embeddings = cfg["max_position_embeddings"]
        self.ffn_inter_dim = cfg["ffn_dim"]
        self.layer_norm_eps = cfg.get("layer_norm_eps", 1e-5)
        # OPT has no RoPE — fill placeholders so shared engine code doesn't crash
        self.rope_theta = None
        self.rope_scaling_factor = 1.0
        # TP placeholders (filled by executor)
        self.rank = None
        self.world_size = None

    def get_kvslot_size(self, extra_layer: bool = False,
                        dtype: torch.dtype = torch.float16) -> int:
        return (2 * (self.num_layers + extra_layer)
                * self.num_kv_heads * self.head_dim) * dtype.itemsize

    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** -0.5

    @staticmethod
    def load_from_model_path(model_path: str) -> "OptModelConfig":
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            return OptModelConfig(json.load(f))
