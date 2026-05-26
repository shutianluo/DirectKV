import time
import torch
import triton
import triton.language as tl
from triton.compiler.compiler import CompiledKernel

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.infer_state import LlamaInferState

# Map from seq_block_size to the compiled kernel, the former must be power of 2
cached_phase1_bin = {}

@triton.jit
def _fwd_paged_attention_phase1(
    mid_o: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim], contiguous. num_seq_blocks = ceil(max_seq_len / seq_block_size)
    mid_o_logexpsum: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks], contiguous
    q: torch.Tensor,    	# [num_decoding_seqs, num_q_heads, head_dim], contiguous
    k: torch.Tensor,		# [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,		# [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    k_cache: torch.Tensor,	# [num_layers, num_blocks, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_layers, num_blocks, num_kv_heads, block_size, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    softmax_scale: tl.float16,
    decoding_seq_lens: torch.Tensor,	# [num_decoding_seqs], contiguous
    seq_ids: torch.Tensor,  # [num_decoding_seqs], contiguous
    num_seq_blocks: int,
    cur_layer: int,

    num_blocks: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_my_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    seq_block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_decoding_seqs, num_q_heads, num_seq_blocks]
    my_batch_id = tl.program_id(0).to(tl.int64)
    my_q_head_id = tl.program_id(1).to(tl.int64)
    my_seq_block_id = tl.program_id(2)
    # num_my_heads = num_q_heads // num_kv_heads
    my_kv_head_id = my_q_head_id // num_my_heads

    my_seq_id = tl.load(seq_ids + my_batch_id)
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    my_start_token_idx = my_seq_block_id * seq_block_size

    if my_start_token_idx >= my_seq_len:
        return

    offs_q = my_batch_id*num_q_heads*head_dim + my_q_head_id*head_dim + tl.arange(0, head_dim)
    my_q = tl.load(q + offs_q) # [head_dim]    

    start_block_idx = my_seq_block_id*(seq_block_size//block_size)
    k_ptrs = k_cache + (cur_layer*num_blocks*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]
    v_ptrs = v_cache + (cur_layer*num_blocks*num_kv_heads+my_kv_head_id)*block_size*head_dim + tl.arange(0, block_size)[:, None]*head_dim + tl.arange(0, head_dim)[None, :]

    max_score = float("-1e20")
    sum_exp = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    # In the following code we deal with the case where the sequence block is
    # the last one in the sequence separately, because:
    #   - The last sequence block may not be a full block, therefore maskings
    #     are needed.
    #   - We can use tl.arange() when the sequence block is not the last one,
    #     leading to better performance.
    if my_start_token_idx + seq_block_size >= my_seq_len:
        # First store the new KV cache
        my_block_pos = (my_seq_len-1) // block_size
        my_block_offset = (my_seq_len-1) % block_size
        my_block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + my_block_pos).to(tl.int64)
        offs_kv = (my_batch_id * num_kv_heads + my_kv_head_id) * head_dim + tl.arange(0, head_dim)
        offs_kvcache = (((cur_layer*num_blocks + my_block_index) * num_kv_heads + my_kv_head_id) * block_size + my_block_offset) * head_dim + tl.arange(0, head_dim)
        tl.store(k_cache + offs_kvcache, tl.load(k + offs_kv))
        tl.store(v_cache + offs_kvcache, tl.load(v + offs_kv))
        
        # The seq block I am processing is the last one in the sequence
        my_num_blocks = tl.cdiv(
            my_seq_len - my_start_token_idx,
            block_size
        )

        for block_i in range(0, my_num_blocks):
            block_idx = start_block_idx + block_i
            block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx).to(tl.int64)
            k_block = tl.load(k_ptrs + block_index*num_kv_heads*block_size*head_dim) # [block_size, head_dim]
            attn_score = tl.sum(my_q[None, :] * k_block, axis=1) # [block_size]
            attn_score = attn_score * softmax_scale
            offs_token = block_i*block_size + my_start_token_idx + tl.arange(0, block_size)
            attn_score = tl.where(offs_token < my_seq_len, attn_score, float('-1e20'))
            v_block = tl.load(v_ptrs + block_index*num_kv_heads*block_size*head_dim) # [block_size, head_dim]
            
            cur_max_score = tl.max(attn_score, axis=0)
            new_max_score = tl.maximum(max_score, cur_max_score)
            exp_attn_score = tl.math.exp2(attn_score - new_max_score)
            old_acc_scale = tl.math.exp2(max_score - new_max_score)

            acc = acc*old_acc_scale + tl.sum(exp_attn_score[:, None]*v_block, axis=0)
            sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=0)
            max_score = new_max_score
    else:
        # The seq block I am processing is NOT the last one in the sequence
        for block_i in tl.static_range(0, seq_block_size // block_size):
            block_idx = start_block_idx + block_i
            block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx).to(tl.int64)
            k_block = tl.load(k_ptrs + block_index*num_kv_heads*block_size*head_dim) # [block_size, head_dim]
            attn_score = tl.sum(my_q[None, :] * k_block, axis=1) # [block_size]
            attn_score = attn_score * softmax_scale
            v_block = tl.load(v_ptrs + block_index*num_kv_heads*block_size*head_dim) # [block_size, head_dim]
            
            cur_max_score = tl.max(attn_score, axis=0)
            new_max_score = tl.maximum(max_score, cur_max_score)
            exp_attn_score = tl.math.exp2(attn_score - new_max_score)
            old_acc_scale = tl.math.exp2(max_score - new_max_score)

            acc = acc*old_acc_scale + tl.sum(exp_attn_score[:, None]*v_block, axis=0)
            sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=0)
            max_score = new_max_score

    offs_mid_o = my_batch_id*num_q_heads*num_seq_blocks*head_dim + my_seq_block_id*head_dim + (my_q_head_id*num_seq_blocks*head_dim) + tl.arange(0, head_dim)
    tl.store(mid_o + offs_mid_o, acc / sum_exp)
    offs_mid_o_logexpsum = my_batch_id*num_q_heads*num_seq_blocks + my_seq_block_id + my_q_head_id*num_seq_blocks
    tl.store(mid_o_logexpsum + offs_mid_o_logexpsum, tl.math.log2(sum_exp) + max_score)   # Here tl.log(sum_exp) + max_score = log(sum(e^{a_i}))


cached_phase2_bin = {}

@triton.jit
def _fwd_paged_attention_phase2(
    mid_o: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim], contiguous
    mid_o_logexpsum: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks], contiguous
    o: torch.Tensor,		# [num_decoding_seqs, num_q_heads, head_dim], contiguous

    decoding_seq_lens: torch.Tensor,	# [num_decoding_seqs], contiguous

    num_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_seq_blocks: tl.constexpr,
    seq_block_size: tl.constexpr,
):
    # grid shape: [num_decoding_seqs, num_q_heads]
    my_batch_id = tl.program_id(0)
    my_q_head_id = tl.program_id(1)

    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    my_num_seq_blocks = tl.cdiv(my_seq_len, seq_block_size)

    sum_exp = 0.0
    max_score = float("-1e20")
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for seq_block_id in range(my_num_seq_blocks):
        offs_mid_o = ((my_batch_id*num_q_heads+my_q_head_id)*num_seq_blocks+seq_block_id)*head_dim + tl.arange(0, head_dim)
        offs_mid_o_logexpsum = (my_batch_id*num_q_heads+my_q_head_id)*num_seq_blocks+seq_block_id
        cur_mid_o = tl.load(mid_o + offs_mid_o)   # [head_dim]
        cur_mid_o_logexpsum = tl.load(mid_o_logexpsum + offs_mid_o_logexpsum)

        new_max_score = tl.maximum(max_score, cur_mid_o_logexpsum)
        old_scale = tl.math.exp2(max_score - new_max_score)
        exp_score = tl.math.exp2(cur_mid_o_logexpsum - new_max_score)
        acc = acc * old_scale + exp_score * cur_mid_o
        sum_exp = sum_exp * old_scale + exp_score
        max_score = new_max_score

    offs_o = (my_batch_id*num_q_heads+my_q_head_id)*head_dim + tl.arange(0, head_dim)
    tl.store(o + offs_o, (acc / sum_exp).to(tl.float16))


def paged_attention(
    q: torch.Tensor,                    # [num_decoding_seqs, num_q_heads, head_dim]
    k: torch.Tensor,                    # [num_decoding_seqs, num_kv_heads, head_dim]
    v: torch.Tensor,                    # [num_decoding_seqs, num_kv_heads, head_dim]
    o: torch.Tensor,     # [num_decoding_seqs, num_q_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    softmax_scale: float,
    block_table: torch.Tensor,
    seq_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    cur_layer: int,
    seq_block_size: int,
    num_seq_blocks: int
):
    start = time.perf_counter()
    assert q.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert o.is_contiguous()

    num_q_heads = q.shape[1]
    head_dim = q.shape[2]
    num_blocks = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[3]
    block_table_width = block_table.shape[1]
    assert seq_block_size % block_size == 0

    num_decoding_seqs = seq_ids.shape[0]


    mid_o = torch.empty((
        num_decoding_seqs,
        num_q_heads,
        num_seq_blocks,
        head_dim
    ), device=q.device, dtype=torch.float32)
    mid_o_logexpsum = torch.empty((
        num_decoding_seqs,
        num_q_heads,
        num_seq_blocks
    ), device=q.device, dtype=torch.float32)

    grid = (num_decoding_seqs, num_q_heads, num_seq_blocks)

    global cached_phase1_bin
    if True: #seq_block_size not in cached_phase1_bin:
        cached_phase1_bin[seq_block_size] = _fwd_paged_attention_phase1[grid](
            mid_o, mid_o_logexpsum,
            q, k, v, k_cache, v_cache,
            block_table,

            # Here we multiply softmax_scale by log2(e) and use `exp2` instead of
            # `exp` because of two reasons:
            # 1. Up to 12 Jun 2024, all NVIDIA GPUs does not have a `exp` instruction
            #    in PTX. When calculating `exp`, they multiply the input by log2(e)
            #    and use `exp2` instead.
            # 2. Some optimizations are disabled while using `exp` in a loop, see
            #    https://github.com/triton-lang/triton/issues/2961
            softmax_scale * 1.442695040888963,
            seq_lens,
            seq_ids,
            num_seq_blocks,
            cur_layer,

            num_blocks,
            num_q_heads,
            num_kv_heads,
            num_q_heads // num_kv_heads,
            block_size,
            head_dim,
            seq_block_size,
            block_table_width,
            num_warps = 1,
            num_stages = 4
        )
    else:
        cached_phase1_bin[seq_block_size][grid](
            mid_o, mid_o_logexpsum,
            q, k, v, k_cache, v_cache,
            block_table,
            softmax_scale * 1.442695040888963,
            seq_lens,
            seq_ids,
            num_seq_blocks,
            cur_layer
        )

    grid = (num_decoding_seqs, num_q_heads, 1)

    global cached_phase2_bin
    if True: #seq_block_size not in cached_phase2_bin:
        cached_phase2_bin[seq_block_size] = _fwd_paged_attention_phase2[grid](
            mid_o, mid_o_logexpsum,
            o,
            seq_lens,
            num_q_heads,
            head_dim,
            num_seq_blocks,
            seq_block_size
        )
    else:
        cached_phase2_bin[seq_block_size][grid](
            mid_o, mid_o_logexpsum,
            o,
            seq_lens
        )
    
    end = time.perf_counter()
    return end - start