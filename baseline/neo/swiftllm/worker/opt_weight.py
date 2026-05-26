import os
import json
import dataclasses
import torch
import safetensors

from swiftllm.opt_model_config import OptModelConfig
from swiftllm.worker.weight import RegisteredWeightItem, WeightBase


class OptTransformerLayerWeight(WeightBase):
    """
    Per-layer weights for an OPT transformer block.

    Naming scheme in HF safetensors:
      model.decoder.layers.N.self_attn_layer_norm.{weight,bias}
      model.decoder.layers.N.self_attn.{q,k,v,out}_proj.{weight,bias}
      model.decoder.layers.N.final_layer_norm.{weight,bias}
      model.decoder.layers.N.fc1.{weight,bias}
      model.decoder.layers.N.fc2.{weight,bias}
    """

    def __init__(self, layer_id: int, model_config: OptModelConfig,
                 dtype: torch.dtype):
        super().__init__(model_config, dtype)
        self.layer_id = layer_id
        H = model_config.hidden_size
        NH = model_config.num_q_heads
        KVH = model_config.num_kv_heads
        D = model_config.head_dim
        F = model_config.ffn_inter_dim
        n = f"model.decoder.layers.{layer_id}"

        # Attention pre-norm (LayerNorm)
        self.register_weight(RegisteredWeightItem("attn_norm_w", f"{n}.self_attn_layer_norm.weight", (H,), (False,), dtype))
        self.register_weight(RegisteredWeightItem("attn_norm_b", f"{n}.self_attn_layer_norm.bias",   (H,), (False,), dtype))

        # QKV projections (with bias, split Q/K/V along head dim on dim-0)
        self.register_weight(RegisteredWeightItem("q_proj",  f"{n}.self_attn.q_proj.weight",  (NH*D, H),  (True,  False), dtype))
        self.register_weight(RegisteredWeightItem("q_bias",  f"{n}.self_attn.q_proj.bias",    (NH*D,),    (True,),        dtype))
        self.register_weight(RegisteredWeightItem("k_proj",  f"{n}.self_attn.k_proj.weight",  (KVH*D, H), (True,  False), dtype))
        self.register_weight(RegisteredWeightItem("k_bias",  f"{n}.self_attn.k_proj.bias",    (KVH*D,),   (True,),        dtype))
        self.register_weight(RegisteredWeightItem("v_proj",  f"{n}.self_attn.v_proj.weight",  (KVH*D, H), (True,  False), dtype))
        self.register_weight(RegisteredWeightItem("v_bias",  f"{n}.self_attn.v_proj.bias",    (KVH*D,),   (True,),        dtype))

        # Output projection (split input along dim-1)
        self.register_weight(RegisteredWeightItem("o_proj", f"{n}.self_attn.out_proj.weight", (H, NH*D),  (False, True),  dtype))
        self.register_weight(RegisteredWeightItem("o_bias", f"{n}.self_attn.out_proj.bias",   (H,),       (False,),       dtype))

        # FFN pre-norm (LayerNorm)
        self.register_weight(RegisteredWeightItem("ffn_norm_w", f"{n}.final_layer_norm.weight", (H,), (False,), dtype))
        self.register_weight(RegisteredWeightItem("ffn_norm_b", f"{n}.final_layer_norm.bias",   (H,), (False,), dtype))

        # FC1 (up-projection, split along dim-0)
        self.register_weight(RegisteredWeightItem("fc1",      f"{n}.fc1.weight", (F, H),   (True,  False), dtype))
        self.register_weight(RegisteredWeightItem("fc1_bias", f"{n}.fc1.bias",   (F,),     (True,),        dtype))

        # FC2 (down-projection, split along dim-1)
        self.register_weight(RegisteredWeightItem("fc2",      f"{n}.fc2.weight", (H, F),   (False, True),  dtype))
        self.register_weight(RegisteredWeightItem("fc2_bias", f"{n}.fc2.bias",   (H,),     (False,),       dtype))

    def _post_process_after_load(self, getter):
        pass  # nothing to fuse for OPT


class OptWeight(WeightBase):
    """
    Global OPT weights: embeddings, final norm, lm_head.
    """

    def __init__(self, model_config: OptModelConfig, dtype: torch.dtype):
        super().__init__(model_config, dtype)
        H = model_config.hidden_size
        V = model_config.vocab_size
        # +2: OPT position embedding table has an offset of 2 (positions 0,1 reserved)
        P = model_config.max_position_embeddings + 2

        self.register_weight(RegisteredWeightItem(
            "embed_tokens", "model.decoder.embed_tokens.weight",
            (V, H), (True, False), dtype))
        self.register_weight(RegisteredWeightItem(
            "embed_positions", "model.decoder.embed_positions.weight",
            (P, H), (False, False), dtype))
        self.register_weight(RegisteredWeightItem(
            "final_norm_w", "model.decoder.final_layer_norm.weight",
            (H,), (False,), dtype))
        self.register_weight(RegisteredWeightItem(
            "final_norm_b", "model.decoder.final_layer_norm.bias",
            (H,), (False,), dtype))
        self.register_weight(RegisteredWeightItem(
            "lm_head", "lm_head.weight",
            (V, H), (True, False), dtype))

        self.layers: list[OptTransformerLayerWeight] = []
        for i in range(model_config.num_layers):
            self.layers.append(
                OptTransformerLayerWeight(i, model_config, dtype))

    def _post_process_after_load(self, getter):
        for layer in self.layers:
            layer.load_weights(getter)


def load_opt_weights(
    model_config: OptModelConfig,
    dtype: torch.dtype,
    model_path: str,
    use_dummy: bool = False,
) -> OptWeight:
    rk = model_config.rank
    ws = model_config.world_size

    if use_dummy:
        assert rk == 0 and ws == 1
        def getter(item: RegisteredWeightItem):
            return torch.empty(
                item.get_real_shape(ws), dtype=item.dtype, device="cuda"
            ).uniform_(-0.001, 0.001)
    else:
        safetensor_files = [n for n in os.listdir(model_path)
                            if n.endswith(".safetensors")]
        safetensor_index_path = os.path.join(
            model_path, "model.safetensors.index.json")
        if os.path.exists(safetensor_index_path):
            with open(safetensor_index_path, encoding="utf-8") as f:
                safetensor_index = json.load(f)["weight_map"]
            safetensor_filename = None
        else:
            assert len(safetensor_files) == 1
            safetensor_index = None
            safetensor_filename = safetensor_files[0]

        def getter(item: RegisteredWeightItem):
            fname = (safetensor_index[item.key]
                     if safetensor_index is not None else safetensor_filename)
            fpath = os.path.join(model_path, fname)
            with safetensors.safe_open(fpath, framework="pt", device="cuda") as f:
                whole = f.get_slice(item.key)
                slices = [
                    slice(rk * size // ws, (rk + 1) * size // ws)
                    if split else slice(0, size)
                    for size, split in item.shape_with_split
                ]
                return whole[slices].to(item.dtype)

    weight = OptWeight(model_config, dtype)
    weight.load_weights(getter)
    return weight
