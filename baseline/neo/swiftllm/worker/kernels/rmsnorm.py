import torch
import triton
import triton.language as tl
import swiftllm_c

@triton.jit
def _fwd_rmsnorm(
	input_and_output: torch.Tensor,	# [num_tokens, hidden_size], contiguous
	weight: torch.Tensor,			# [hidden_size]
	eps: float,

	hidden_size: tl.constexpr
):
	# grid shape: [num_tokens]
	my_token_id = tl.program_id(0)
	input_and_output += my_token_id * hidden_size	# [hidden_size]

	offs = tl.arange(0, hidden_size)
	x = tl.load(input_and_output+offs).to(tl.float32)
	variance = tl.sum(x*x, axis=0) / hidden_size
	rstd = 1 / tl.sqrt(variance + eps)

	w = tl.load(weight+offs).to(tl.float32)
	x = x*rstd*w
	tl.store(input_and_output+offs, x.to(tl.float16))

def rmsnorm_inplace(
	input_and_output: torch.Tensor,	# [num_tokens, hidden_size]
	weight: torch.Tensor,
	eps: float
):
	grid = (input_and_output.shape[0], )
	_fwd_rmsnorm[grid](
		input_and_output,
		weight,
		eps,
		input_and_output.shape[1]
	)

@triton.jit
def _fwd_fused_add_rmsnorm(
	input_and_output: torch.Tensor,	# [num_tokens, hidden_size], contiguous
	residual_io: torch.Tensor,		# [num_tokens, hidden_size], contiguous
	weight: torch.Tensor,			# [hidden_size]
	eps: float,

	hidden_size: tl.constexpr
):
	# grid shape: [num_tokens]
	my_token_id = tl.program_id(0)
	input_and_output += my_token_id * hidden_size	# [hidden_size]
	residual_io += my_token_id * hidden_size

	offs = tl.arange(0, hidden_size)
	x = tl.load(input_and_output+offs)
	r = tl.load(residual_io+offs)
	x += r
	tl.store(residual_io+offs, x)

	x = x.to(tl.float32)
	variance = tl.sum(x*x, axis=0) / hidden_size
	rstd = 1 / tl.sqrt(variance + eps)

	w = tl.load(weight+offs).to(tl.float32)
	x = x*rstd*w
	tl.store(input_and_output+offs, x.to(tl.float16))

def fused_add_rmsnorm_inplace(
	input_and_output: torch.Tensor,	# [num_tokens, hidden_size]
	residual_io: torch.Tensor,
	weight: torch.Tensor,
	eps: float
):
	"""
	Perform fused add & rmsnorm

	This function accepts input_and_output (x), residual_io (r), and weight(w)
	as inputs, set r = x+r, and x = rms_norm(x+r, w)
	"""
	assert input_and_output.is_contiguous()
	assert residual_io.is_contiguous()
	assert weight.is_contiguous()
	grid = (input_and_output.shape[0], )
	_fwd_fused_add_rmsnorm[grid](
		input_and_output,
		residual_io,
		weight,
		eps,
		input_and_output.shape[1]
	)

if __name__ == "__main__":
	x = torch.randn(10, 128, dtype=torch.float16, device="cuda")
	y = x.clone()
	r = torch.randn(10, 128, dtype=torch.float16, device="cuda")
	s = r.clone()
	w = torch.randn(1024, dtype=torch.float16, device="cuda")
	eps = 1e-6

	fused_add_rmsnorm_inplace(x, r, w, eps)
	swiftllm_c.fused_add_rmsnorm_inplace(y, s, w, eps)

	assert torch.allclose(x, y, atol=1e-5), "Mismatch x, y"
	assert torch.allclose(r, s, atol=1e-5), "Mismatch r, s"
