import dataclasses
import torch

@dataclasses.dataclass
class LlamaInferState:
    batch_size: int
    num_tokens: int

    gpu_seq_ids: torch.Tensor   # [batch_size]
    softmax_scale: float    # Equal to 1/sqrt(head_dim)

    num_prefill_seqs: int
    num_prefill_tokens: int
    prefill_seq_lens: torch.Tensor # [batch_size]
    prefill_seq_start_locs: torch.Tensor # [batch_size]
    prefill_seq_start_locs_with_end: torch.Tensor # [batch_size+1], = prefill_seq_start_locs + [num_prefill_tokens]
    max_prefill_len: int

    gpu_num_decoding_seqs: int
    gpu_decoding_seq_lens: torch.Tensor # [batch_size]

    cpu_num_decoding_seqs: int
    cpu_seq_ids: torch.Tensor
    cpu_decoding_seq_lens: torch.Tensor

    @property
    def gpu_token_end(self) -> int:
        return self.num_tokens - self.cpu_num_decoding_seqs

    seq_block_size: int
    num_seq_blocks: int

    position_cos: torch.Tensor	# [num_tokens, hidden_size]
    position_sin: torch.Tensor	# [num_tokens, hidden_size]

    src_block_ids: list[int]
    dst_block_ids: list[int]
