"""
BlockManager and Swapper

Contains initalization and transition logics for the KV cache.
"""

import torch
import swiftllm_c
from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig

class Swapper:
    """
    Swapper - Manage the swapping of sequences in and out of the model

    This manager is responsible for swapping sequences in and out of the model.
    It maintains the block manager, and provides methods to swap sequences in
    and out.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: LlamaModelConfig
    ):
        self.engine_config = engine_config
        self.model_config = model_config

        num_q_heads = model_config.num_q_heads // model_config.world_size
        num_kv_heads = model_config.num_kv_heads // model_config.world_size

        # Initialize KV cache
        kvcache_shape = (
            model_config.num_layers + engine_config.extra_layer_for_cprf,
            engine_config.num_gpu_blocks,
            num_kv_heads,
            engine_config.block_size,
            model_config.head_dim
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")

        # Initialize KV swap space
        kvswap_shape = (
            model_config.num_layers,
            engine_config.num_cpu_blocks,
            num_kv_heads,
            engine_config.block_size,
            model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu", pin_memory=True)

        # Initialize CPU QKV buffer
        qo_cpu_shape = (engine_config.max_batch_size, num_q_heads, model_config.head_dim)
        kv_cpu_shape = (engine_config.max_batch_size, num_kv_heads, model_config.head_dim)
        self.q_cpu = torch.zeros(qo_cpu_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        self.k_cpu = torch.zeros(kv_cpu_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        self.v_cpu = torch.zeros(kv_cpu_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        # We store float32 tensors for the output, but convert them to float16 after copying back to GPU
        self.o_cpu = torch.zeros(qo_cpu_shape, dtype=torch.float32, device="cpu", pin_memory=True)

        self.gpu_block_table = torch.zeros(
            (engine_config.max_seqs_in_block_table, engine_config.max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda"
        )
        self.cpu_block_table = torch.zeros(
            (engine_config.max_seqs_in_block_table, engine_config.max_blocks_per_seq),
            dtype=torch.int32,
            device="cpu"
        )

    
    def swap_blocks(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_swap_out: bool,
        gpu_layer: int,
        cpu_layer: int
    ):
        """
        Swap blocks between the GPU and CPU, the physical indexes of the blocks are given.
        """
        # pylint: disable=too-many-arguments, c-extension-no-member
        assert len(src_block_ids) == len(dst_block_ids), "Length mismatch between src_block_ids and dst_block_ids"
        if not src_block_ids:
            return
        swiftllm_c.swap_blocks(
            src_block_ids,
            dst_block_ids,
            is_swap_out,
            gpu_layer,
            cpu_layer,

            self.k_cache, self.v_cache,
            self.k_swap, self.v_swap
        )

    @torch.inference_mode()
    def set_block_tables(
        self,
        mappings: tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]     
    ):
        """
        Establish new mappings in the block tables
        """
        (gpu_vids, gpu_pids), (cpu_vids, cpu_pids) = mappings
        if gpu_vids:
            self.gpu_block_table.view(-1)[gpu_vids] = torch.tensor(gpu_pids, dtype=torch.int32, device="cuda")
        if cpu_vids:
            self.cpu_block_table.view(-1)[cpu_vids] = torch.tensor(cpu_pids, dtype=torch.int32, device="cpu")
