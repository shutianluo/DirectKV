"""
OptModel — OPT inference model for swiftllm.

Mirrors LlamaModel's public interface (same __init__ args, init_kvcache_and_swap,
do_one_iteration) so it can be dropped into SingleProcExecutor / RayExecutor
without any other executor changes.
"""

import itertools

import torch
import torch.nn.functional as F
import torch.distributed as dist
import ray

from swiftllm.engine_config import EngineConfig
from swiftllm.opt_model_config import OptModelConfig
from swiftllm.worker.opt_weight import OptWeight, load_opt_weights
from swiftllm.worker.buffer import ModelForwardBuffers
from swiftllm.worker.block_swapper import Swapper
from swiftllm.structs import Request, SubBatch
from swiftllm.worker.kernels.linear import linear

from swiftllm.worker.layers.opt_transformer_layer import OptTransformerLayer

# ------------------------------------------------------------------
# Pre-layer: token embed + learned absolute position embed
# ------------------------------------------------------------------

class OptPreLayer:
    def __init__(self, model_config: OptModelConfig, weights: OptWeight):
        self.model_config = model_config
        self.weights = weights

    def forward(self, input_ids: list[int], position_ids: list[int]) -> torch.Tensor:
        ids_gpu = torch.tensor(input_ids,     dtype=torch.long, device="cuda")
        pos_gpu = torch.tensor(position_ids,  dtype=torch.long, device="cuda")
        # OPT position table has a hard offset of 2 (indices 0,1 reserved)
        tok_emb = F.embedding(ids_gpu,      self.weights.embed_tokens)
        pos_emb = F.embedding(pos_gpu + 2,  self.weights.embed_positions)
        return (tok_emb + pos_emb).half()


# ------------------------------------------------------------------
# Post-layer: final LayerNorm + lm_head
# ------------------------------------------------------------------

class OptPostLayer:
    def __init__(self, model_config: OptModelConfig, weights: OptWeight):
        self.model_config = model_config
        self.weights = weights

    def forward(
        self,
        batches: list[SubBatch],
        input_embeds: torch.Tensor,   # [total_tokens, H]
        residual_buf: torch.Tensor,   # [total_tokens, H]
    ) -> list[int]:
        offs = 0
        last_token_indices = None
        for batch in batches:
            idx = batch.last_token_indices + offs
            last_token_indices = idx if last_token_indices is None else \
                torch.cat((last_token_indices, idx))
            offs += batch.iter_width

        input_embeds = input_embeds[last_token_indices, :]
        residual_buf = residual_buf[last_token_indices, :]

        if self.model_config.world_size > 1:
            dist.all_reduce(input_embeds)

        # Final LayerNorm
        input_embeds.add_(residual_buf)
        input_embeds = F.layer_norm(
            input_embeds.float(),
            [self.model_config.hidden_size],
            self.weights.final_norm_w.float(),
            self.weights.final_norm_b.float(),
            self.model_config.layer_norm_eps,
        ).half()

        logits = linear(input_embeds, self.weights.lm_head)

        if self.model_config.world_size > 1:
            gather_list = (
                [torch.zeros_like(logits) for _ in range(self.model_config.world_size)]
                if self.model_config.rank == 0 else None
            )
            dist.gather(logits, gather_list)
            if self.model_config.rank == 0:
                logits = torch.cat(gather_list, dim=1)

        return torch.argmax(logits, dim=1).tolist() if self.model_config.rank == 0 else []


# ------------------------------------------------------------------
# OptModel
# ------------------------------------------------------------------

class OptModel:
    """
    OPT inference model with swiftllm paged-attention (GPU + CPU via pacpu).
    Public interface is identical to LlamaModel.
    """

    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: OptModelConfig,
        rank: int,
    ):
        self.engine_config = engine_config
        self.model_config = model_config

        model_config.rank = rank
        model_config.world_size = engine_config.tensor_parallel_degree

        if engine_config.library_path:
            torch.ops.load_library(engine_config.library_path)

        self.cpu_communication_stream = torch.cuda.Stream()

        self.weight: OptWeight = load_opt_weights(
            model_config, torch.float16,
            engine_config.model_path, engine_config.use_dummy,
        )

        self.buffer = ModelForwardBuffers(engine_config, model_config)

        self.pre_layer = OptPreLayer(model_config, self.weight)
        self.transformer_layers = [
            OptTransformerLayer(
                model_config, engine_config,
                self.weight.layers[layer_id],
                self.weight.layers[(layer_id + 1) % model_config.num_layers],
                self.cpu_communication_stream,
                layer_id,
            )
            for layer_id in range(model_config.num_layers)
        ]
        self.post_layer = OptPostLayer(model_config, self.weight)

        self.swapper = None
        self.perf_results = []

    @torch.inference_mode()
    def init_kvcache_and_swap(self, engine_config: EngineConfig):
        self.engine_config.num_cpu_blocks = engine_config.num_cpu_blocks
        self.engine_config.num_gpu_blocks = engine_config.num_gpu_blocks
        self.swapper = Swapper(self.engine_config, self.model_config)
        for layer in self.transformer_layers:
            layer.set_swapper(self.swapper)

    # ------------------------------------------------------------------
    # Compute OPT position indices for a flat list of tokens across batches
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_position_ids(batches: list[SubBatch]) -> list[int]:
        pos = []
        for batch in batches:
            # prefill tokens: positions 0..prompt_len-1
            for req in batch.all_reqs[:batch.num_prefs]:
                pos.extend(range(req.prompt_len))
            # decode tokens: position = current seq_len - 1
            for req in batch.all_reqs[batch.num_prefs:]:
                pos.append(req.seq_len - 1)
        return pos

    def _prepare_inputs(self, batches: list[SubBatch]):
        for batch in batches:
            batch.prgd_seq_ids = torch.tensor(
                batch.seq_ids_list[:batch.num_prgds], dtype=torch.int32, device="cuda")
            batch.prgd_seq_lens = torch.tensor(
                batch.seq_lens_list[:batch.num_prgds], dtype=torch.int32, device="cuda")
            batch.pref_st_locs_we = torch.tensor(
                [0] + list(itertools.accumulate(
                    batch.seq_lens_list[:batch.num_prefs])),
                dtype=torch.int32, device="cuda",
            )
            # No RoPE: leave cos/sin as None (OptTransformerLayer ignores them)
            batch.position_cos = None
            batch.position_sin = None

            batch.attn_out_buf = torch.zeros(
                (batch.iter_width,
                 self.model_config.hidden_size // self.model_config.world_size),
                dtype=torch.float16, device="cuda",
            )
            batch.residual_buf = torch.zeros(
                (batch.iter_width, self.model_config.hidden_size),
                dtype=torch.float16, device="cuda",
            )
            batch.last_token_indices = torch.cat([
                batch.pref_st_locs_we[1:] - 1,
                torch.arange(batch.sum_pref_toks, batch.iter_width,
                             dtype=torch.int32, device="cuda"),
            ])

        self.buffer.alloc_for_batches(batches)

    def _forward_sequential(self, batch: SubBatch,
                             embeddings: torch.Tensor) -> torch.Tensor:
        torch.cuda.current_stream().wait_stream(self.cpu_communication_stream)
        for layer in self.transformer_layers:
            embeddings = layer.forward(batch, embeddings)
        return embeddings

    @torch.inference_mode()
    def _forward_batches(self, batches: list[SubBatch]) -> list[int]:
        self._prepare_inputs(batches)

        input_ids   = sum([Request.get_input_tokens(b.all_reqs) for b in batches], [])
        position_ids = self._compute_position_ids(batches)

        embeddings = self.pre_layer.forward(input_ids, position_ids)

        # Sequential only (pipeline mode not implemented for OPT)
        embeddings = self._forward_sequential(batches[0], embeddings)

        return self.post_layer.forward(
            batches, embeddings, self.buffer.cur_residual_buf)

    def do_one_iteration(
        self,
        batches: list[SubBatch],
        mappings,
        swappings,
        is_swap_out: bool = False,
    ) -> list[int]:
        if self.swapper is not None:
            self.swapper.set_block_tables(mappings)

        if swappings[0]:
            with torch.cuda.stream(self.cpu_communication_stream):
                for layer_id in range(self.model_config.num_layers):
                    self.swapper.swap_blocks(
                        *swappings, is_swap_out, layer_id, layer_id)

        return self._forward_batches(batches)

    # Stubs so the engine doesn't crash when calling these
    def turn_on_perf_monitor(self):
        pass

    def turn_off_perf_monitor_and_flush_results(self):
        ret = self.perf_results
        self.perf_results = []
        return ret


@ray.remote(num_cpus=8, num_gpus=1)
class RemoteOptModel(OptModel):
    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: OptModelConfig,
        rank: int,
    ):
        dist.init_process_group(
            backend="nccl",
            world_size=engine_config.tensor_parallel_degree,
            rank=rank,
        )
        super().__init__(engine_config, model_config, rank)
