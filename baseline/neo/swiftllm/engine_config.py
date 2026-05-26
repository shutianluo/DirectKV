"""
Configuration for the SwiftLLM engine.
"""

import dataclasses
import argparse

@dataclasses.dataclass
class EngineConfig:
    """
    Configuration for the SwiftLLM engine.
    """
    
    # Model loading parameters
    model_path: str
    use_dummy: bool

    # PagedAttention-related parameters
    block_size: int
    gpu_mem_utilization: float
    num_gpu_blocks_override: int # -1 for not overriding
    swap_space: int
    max_seqs_in_block_table: int
    max_blocks_per_seq: int

    # Scheduling-related parameters
    max_batch_size: int
    max_tokens_in_batch: int

    # External paths
    library_path: str
    profile_result_path: str

    # Switches
    extra_layer_for_cprf: bool = False  # Fixed after initialization
    disable_partial_offl: bool = False   # Fixed after initialization
    monitor_performance: bool = False   # Can be altered while running
    always_use_gpu: bool = False        # Can be altered while running

    # Parallel parameter
    tensor_parallel_degree: int = 1

    # Derived parameters
    num_cpu_blocks: int = -1
    num_gpu_blocks: int = -1

    @property
    def max_seq_len(self) -> int:
        """
        Maximum sequence length in tokens
        """
        return self.block_size * self.max_blocks_per_seq
    
    @property
    def max_gpu_tokens(self) -> int:
        """
        Maximum number of tokens that can be stored in the GPU
        """
        return self.block_size * self.num_gpu_blocks
    
    @property
    def max_cpu_tokens(self) -> int:
        """
        Maximum number of tokens that can be stored in the CPU
        """
        return self.block_size * self.num_cpu_blocks

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        """
        Add CLI arguments for the engine configuration
        """
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the model directory (currently SwiftLLM does not support downloading from HuggingFace, so please download in advance)",
        )
        parser.add_argument(
            "--use-dummy",
            action="store_true",
            help="Use dummy weights (mainly for profiling)",
        )

        parser.add_argument(
            "--block-size",
            type=int,
            default=16,
            help="Block size for PagedAttention",
        )
        parser.add_argument(
            "--gpu-mem-utilization",
            type=float,
            default=0.99,
            help="Fraction of GPU memory to be used",
        )
        parser.add_argument(
            "--num-gpu-blocks-override",
            type=int,
            default=-1,
            help="Override the number of GPU blocks",
        )
        parser.add_argument(
            "--swap-space",
            type=int,
            default=20,
            help="Swap space in GB",
        )
        parser.add_argument(
            "--max-seqs-in-block-table",
            type=int,
            default=768,
            help="Maximum number of sequences in the block table",
        )
        parser.add_argument(
            "--max-blocks-per-seq",
            type=int,
            default=512,
            help="Maximum number of blocks per sequence",
        )

        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=512,
            help="Maximum batch size",
        )
        parser.add_argument(
            "--max-tokens-in-batch",
            type=int,
            default=3072,
            help="Maximum number of tokens in a batch",
        )

        parser.add_argument(
            "--library-path",
            type=str,
            help="Path to the external library",
        )
        parser.add_argument(
            "--profile-result-path",
            type=str,
            help="Path to the profiling results",
        )
        parser.add_argument(
            "--tensor-parallel-degree",
            type=int,
            default=1,
            help="Degree of tensor parallelism",
        )
        parser.add_argument(
            "--disable-partial-offl",
            action="store_true",
            help="Disable partial offloading",
        )
        parser.add_argument(
            "--always-use-gpu",
            action="store_true",
            help="Always use GPU",
        )
        parser.add_argument(
            "--extra-layer-for-cprf",
            action="store_true",
            help="Use an extra layer for CPRF",
        )
