import time
import torch
import triton
import triton.language as tl

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.utils import cdiv

@triton.jit
def _fwd_kvcache_mgmt_prefill_kernel(
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    seq_ids: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_start_locs: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_lens: torch.Tensor,  # [num_prefill_seqs], contiguous
    cur_layer: int,

    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_prefill_seqs, cdiv(max_prefill_len, block_size)]
    my_batch_id = tl.program_id(0)
    my_block_id = tl.program_id(1)
    my_seq_len = tl.load(prefill_seq_lens + my_batch_id)
    my_seq_start_loc = tl.load(prefill_seq_start_locs + my_batch_id)
    if my_block_id*block_size >= my_seq_len:
        return
    
    my_token_range = tl.arange(0, block_size).to(tl.int64) + my_block_id*block_size + my_seq_start_loc
    my_seq_id = tl.load(seq_ids + my_batch_id)
    my_block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + my_block_id).to(tl.int64)

    offs_kv = (my_token_range*num_kv_heads*head_dim).to(tl.int64)[:, None, None] + (tl.arange(0, num_kv_heads)*head_dim)[None, :, None] + tl.arange(0, head_dim)[None, None, :]
    offs_kvcache = (my_block_index*num_layers+cur_layer)*num_kv_heads*block_size*head_dim + \
        (tl.arange(0, num_kv_heads)*block_size*head_dim)[None, :, None] + \
        (tl.arange(0, block_size)*head_dim)[:, None, None] + \
        tl.arange(0, head_dim)[None, None, :]
    
    mask = (my_token_range < my_seq_len + my_seq_start_loc)[:, None, None]
    tl.store(k_cache + offs_kvcache, tl.load(k + offs_kv, mask=mask), mask=mask)
    tl.store(v_cache + offs_kvcache, tl.load(v + offs_kv, mask=mask), mask=mask)

def store_kvcache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_ids: torch.Tensor,
    prefill_seq_start_locs: torch.Tensor,
    prefill_seq_lens: torch.Tensor,
    cur_layer: int,
    max_prefill_len: int
):
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert seq_ids.is_contiguous()

    num_layers = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[3]
    head_dim = k_cache.shape[4]
    block_table_width = block_table.shape[1]
    num_prefill_seqs = seq_ids.shape[0]

    grid = (num_prefill_seqs, cdiv(max_prefill_len, block_size))
    _fwd_kvcache_mgmt_prefill_kernel[grid](
        k_cache, v_cache,
        k, v,
        block_table,
        seq_ids,
        prefill_seq_start_locs, 
        prefill_seq_lens,
        cur_layer,
        num_layers, num_kv_heads, block_size, head_dim, block_table_width
    )
