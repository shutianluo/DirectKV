import os
import json
import torch

class LlamaModelConfig:
    """
    The configuration of a LLaMA model (including LLaMA 1/2/3).
    """
    
    def __init__(
        self,
        model_config: dict
    ):
        """
        Initialize a LLaMA model configuration from a dict, which should be generated
        from a huggingface transformers config.json file.
        """
        
        assert model_config["model_type"] == "llama"
        self.num_layers = model_config["num_hidden_layers"]
        self.num_q_heads = model_config["num_attention_heads"]
        self.num_kv_heads = model_config.get("num_key_value_heads", self.num_q_heads)
        self.hidden_size = model_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_q_heads
        self.vocab_size = model_config["vocab_size"]
        self.max_position_embeddings = model_config["max_position_embeddings"]
        self.ffn_inter_dim = model_config["intermediate_size"]
        self.rotary_base = model_config.get("rope_theta", model_config.get("rotary_base", 10000))
        self.rms_norm_eps = model_config["rms_norm_eps"]
        self.rope_theta = model_config.get("rope_theta", 10000)
        rope_scaling = model_config.get("rope_scaling", None)
        if rope_scaling is None:
           self.rope_scaling_factor = 1.0
        else:
            self.rope_scaling_factor = 1.0
            # Here we use 1.0 for simplicity, but in practice it should be determined by the model_config["rope_scaling"] dictionary
        assert model_config["hidden_act"] == "silu"

        self.rank = None
        self.world_size = None

    def get_kvslot_size(self, extra_layer: bool = False, dtype: torch.dtype = torch.float16) -> int:
        """
        Get the size of one kv slot (the kv cache of one token) (in bytes)
        """
        return (2 * (self.num_layers + extra_layer) * self.num_kv_heads * self.head_dim) * dtype.itemsize

    @property
    def softmax_scale(self) -> float:
        """
        Get the scale of the softmax function
        """
        return self.head_dim ** -0.5
    
    @staticmethod
    def load_from_model_path(model_path: str) -> "LlamaModelConfig":
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            model_config_dict = json.loads(f.read())
        return LlamaModelConfig(model_config_dict)


def load_model_config(model_path: str):
    """
    Auto-detect model_type from config.json and return the appropriate config object.
    Returns LlamaModelConfig for llama models, OptModelConfig for opt models.
    """
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "")
    if model_type == "opt":
        from swiftllm.opt_model_config import OptModelConfig  # avoid circular import
        return OptModelConfig(cfg)
    return LlamaModelConfig(cfg)
