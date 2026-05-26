import torch
import triton
import triton.language as tl
import swiftllm_c

from swiftllm.worker.infer_state import LlamaInferState

@triton.jit
def _fwd_rotary_embedding(
	q: torch.Tensor,	# [num_tokens, num_q_heads, head_dim]
	k: torch.Tensor,	# [num_tokens, num_k_heads, head_dim]
	cos_table: torch.Tensor,	# [num_tokens, head_dim//2]
	sin_table: torch.Tensor,	# [num_tokens, head_dim//2]

	num_q_heads: tl.constexpr,
	num_kv_heads: tl.constexpr,
	gqa_group_size: tl.constexpr,	# = num_q_heads / num_kv_heads
	head_dim: tl.constexpr
):
	# grid: [num_tokens, num_kv_heads]
	my_token_id = tl.program_id(0)
	my_kv_head = tl.program_id(1)

	q += my_token_id*num_q_heads*head_dim + my_kv_head*gqa_group_size*head_dim	# [gqa_group_size, head_dim]
	k += my_token_id*num_kv_heads*head_dim + my_kv_head*head_dim	# [head_dim]

	offs0 = tl.arange(0, head_dim//2)
	offs1 = tl.arange(head_dim//2, head_dim)

	cos = tl.load(cos_table + my_token_id*(head_dim//2) + offs0)
	sin = tl.load(sin_table + my_token_id*(head_dim//2) + offs0)

	offs_q0 = (tl.arange(0, gqa_group_size)*head_dim)[:, None] + offs0[None, :]
	offs_q1 = (tl.arange(0, gqa_group_size)*head_dim)[:, None] + offs1[None, :]
	q0 = tl.load(q + offs_q0)
	q1 = tl.load(q + offs_q1)
	tl.store(q + offs_q0, q0*cos - q1*sin)
	tl.store(q + offs_q1, q0*sin + q1*cos)

	k0 = tl.load(k + offs0)
	k1 = tl.load(k + offs1)
	tl.store(k + offs0, k0*cos - k1*sin)
	tl.store(k + offs1, k0*sin + k1*cos)

def rotary_embedding_inplace(
	q: torch.Tensor,	# [num_tokens, num_q_heads, head_dim]
	k: torch.Tensor,	# [num_tokens, num_k_heads, head_dim]
	sin_table: torch.Tensor,	# [num_tokens, head_dim//2]
	cos_table: torch.Tensor	  # [num_tokens, head_dim//2]
):
	num_tokens = q.shape[0]
	num_q_heads = q.shape[1]
	num_kv_heads = k.shape[1]
	head_dim = k.shape[2]
	grid = (num_tokens, num_kv_heads)
	_fwd_rotary_embedding[grid](
		q, k,
		cos_table, sin_table,
		num_q_heads, num_kv_heads, num_q_heads//num_kv_heads, head_dim
	)
	
if __name__ == '__main__':
	q0 = torch.randn(4, 32, 128, dtype=torch.float16, device='cuda')
	q1 = q0.clone()
	k0 = torch.randn(4, 32, 128, dtype=torch.float16, device='cuda')
	k1 = k0.clone()
	sin_table = torch.randn(4, 64, dtype=torch.float16, device='cuda')
	cos_table = torch.randn(4, 64, dtype=torch.float16, device='cuda')

	rotary_embedding_inplace(q0, k0, sin_table, cos_table)
	swiftllm_c.rotary_embedding_inplace(q1, k1, sin_table, cos_table)
	print(q0[0,0])
	print(q1[0,0])

	assert torch.allclose(q0, q1, atol=1e-6), "q0 != q1"
	assert torch.allclose(k0, k1, atol=1e-6), "k0 != k1"
