"""
sgl_kernel stub — PyTorch 2.7 ABI compatibility shim for DirectKV AE.

sgl_kernel 0.3.21 was compiled against c10::cuda::SetDevice(int,bool) but
PyTorch 2.7 only exports SetDevice(int).  This stub replaces the broken
__init__.py so sglang's import-time requirements are satisfied.

Any kernel that is *actually called at runtime* raises NotImplementedError.
For the DirectKV microbenchmarks and correctness tests, no sgl_kernel
kernels are invoked.
"""

import torch as _torch
import torch.nn.functional as _F


def _stub(name):
    def _f(*a, **kw):
        raise NotImplementedError(
            f"sgl_kernel.{name} is not available: the sgl_kernel binary is "
            f"ABI-incompatible with torch {_torch.__version__}. "
            f"This is not needed for DirectKV microbenchmarks."
        )
    _f.__name__ = name
    return _f


class _StubMod:
    def __init__(self, name):
        self._name = name
    def __getattr__(self, attr):
        return _stub(f"{self._name}.{attr}")


# ── PyTorch fallbacks for elementwise ops used during serving ─────────────────

def rmsnorm(x, weight, eps):
    """RMS LayerNorm — writes result to a new tensor (returned)."""
    orig_dtype = x.dtype
    x = x.to(_torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * _torch.rsqrt(variance + eps)
    return (weight * x).to(orig_dtype)


def fused_add_rmsnorm(x, residual, weight, eps):
    """Fused add + RMS norm, in-place on x and residual."""
    residual.add_(x)
    orig_dtype = residual.dtype
    r = residual.to(_torch.float32)
    variance = r.pow(2).mean(-1, keepdim=True)
    r = r * _torch.rsqrt(variance + eps)
    x.copy_((weight * r).to(orig_dtype))


def silu_and_mul(x, out):
    """SwiGLU: silu(x[:d]) * x[d:] written to out."""
    d = x.shape[-1] // 2
    out.copy_(_F.silu(x[..., :d]) * x[..., d:])


def gelu_and_mul(x, out):
    d = x.shape[-1] // 2
    out.copy_(_F.gelu(x[..., :d]) * x[..., d:])


def gelu_tanh_and_mul(x, out):
    d = x.shape[-1] // 2
    out.copy_(_F.gelu(x[..., :d], approximate="tanh") * x[..., d:])


def gelu_quick(x, out):
    out.copy_(_F.silu(x))


def gemma_rmsnorm(x, weight, eps):
    return rmsnorm(x, weight + 1.0, eps)


def gemma_fused_add_rmsnorm(x, residual, weight, eps):
    fused_add_rmsnorm(x, residual, weight + 1.0, eps)


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox):
    """Apply rotary embeddings in-place. positions: (bs,). q/k: (total_tokens, nheads*head_size)."""
    rot_dim = cos_sin_cache.shape[-1] // 2
    cos = cos_sin_cache[positions, :rot_dim].to(query.dtype)  # (bs, rot_dim)
    sin = cos_sin_cache[positions, rot_dim:].to(query.dtype)  # (bs, rot_dim)

    def _apply_inplace(t, nheads):
        t3 = t.view(-1, nheads, head_size).clone()  # clone to avoid aliasing
        t_rot = t3[..., :rot_dim].clone()
        t1 = t_rot[..., :rot_dim // 2]
        t2 = t_rot[..., rot_dim // 2:]
        t_rot_new = _torch.cat([-t2, t1], dim=-1)
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        t3[..., :rot_dim] = t_rot * c + t_rot_new * s
        t.copy_(t3.reshape_as(t))

    nq = query.shape[-1] // head_size
    nk = key.shape[-1] // head_size
    _apply_inplace(query, nq)
    _apply_inplace(key, nk)


def apply_rope_with_cos_sin_cache_inplace(positions, query, key, head_size, cos_sin_cache, is_neox=True):
    rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)


# ── version ──────────────────────────────────────────────────────────────────
__version__ = "0.0.0+directkv_stub"

# ── attention / merge state ───────────────────────────────────────────────────
merge_state = _stub("merge_state")
merge_state_v2 = _stub("merge_state_v2")
cutlass_mla_decode = _stub("cutlass_mla_decode")
cutlass_mla_get_workspace_size = _stub("cutlass_mla_get_workspace_size")

# ── elementwise ──────────────────────────────────────────────────────────────
FusedSetKVBufferArg = _stub("FusedSetKVBufferArg")
# apply_rope_with_cos_sin_cache_inplace — implemented above
concat_mla_absorb_q = _stub("concat_mla_absorb_q")
concat_mla_k = _stub("concat_mla_k")
copy_to_gpu_no_ce = _stub("copy_to_gpu_no_ce")
downcast_fp8 = _stub("downcast_fp8")
# fused_add_rmsnorm — implemented above
# gelu_and_mul — implemented above
# gelu_tanh_and_mul — implemented above
# gemma_fused_add_rmsnorm — implemented above
# gemma_rmsnorm — implemented above
# rmsnorm — implemented above
# rotary_embedding — implemented above
# silu_and_mul — implemented above
timestep_embedding = _stub("timestep_embedding")
# gelu_quick — implemented above

# ── gemm ─────────────────────────────────────────────────────────────────────
awq_dequantize = _stub("awq_dequantize")
bmm_fp8 = _stub("bmm_fp8")
cutlass_scaled_fp4_mm = _stub("cutlass_scaled_fp4_mm")
dsv3_fused_a_gemm = _stub("dsv3_fused_a_gemm")
dsv3_router_gemm = _stub("dsv3_router_gemm")
fp8_blockwise_scaled_mm = _stub("fp8_blockwise_scaled_mm")
fp8_scaled_mm = _stub("fp8_scaled_mm")
gptq_gemm = _stub("gptq_gemm")
gptq_marlin_gemm = _stub("gptq_marlin_gemm")
gptq_shuffle = _stub("gptq_shuffle")
int8_scaled_mm = _stub("int8_scaled_mm")
qserve_w4a8_per_chn_gemm = _stub("qserve_w4a8_per_chn_gemm")
qserve_w4a8_per_group_gemm = _stub("qserve_w4a8_per_group_gemm")
scaled_fp4_experts_quant = _stub("scaled_fp4_experts_quant")
scaled_fp4_grouped_quant = _stub("scaled_fp4_grouped_quant")
scaled_fp4_quant = _stub("scaled_fp4_quant")
sgl_per_tensor_quant_fp8 = _stub("sgl_per_tensor_quant_fp8")
sgl_per_token_group_quant_8bit = _stub("sgl_per_token_group_quant_8bit")
sgl_per_token_group_quant_fp8 = _stub("sgl_per_token_group_quant_fp8")
sgl_per_token_group_quant_int8 = _stub("sgl_per_token_group_quant_int8")
sgl_per_token_quant_fp8 = _stub("sgl_per_token_quant_fp8")
shuffle_rows = _stub("shuffle_rows")
silu_and_mul_scaled_fp4_grouped_quant = _stub("silu_and_mul_scaled_fp4_grouped_quant")

# ── moe ──────────────────────────────────────────────────────────────────────
apply_shuffle_mul_sum = _stub("apply_shuffle_mul_sum")
cutlass_fp4_group_mm = _stub("cutlass_fp4_group_mm")
cutlass_w4a8_moe_mm = _stub("cutlass_w4a8_moe_mm")
fp8_blockwise_scaled_grouped_mm = _stub("fp8_blockwise_scaled_grouped_mm")
fused_marlin_moe = _stub("fused_marlin_moe")
fused_qk_norm_rope = _stub("fused_qk_norm_rope")
get_cutlass_w4a8_moe_mm_data = _stub("get_cutlass_w4a8_moe_mm_data")
kimi_k2_moe_fused_gate = _stub("kimi_k2_moe_fused_gate")
moe_align_block_size = _stub("moe_align_block_size")
moe_fused_gate = _stub("moe_fused_gate")
moe_sum = _stub("moe_sum")
moe_sum_reduce = _stub("moe_sum_reduce")
moe_wna16_marlin_gemm = _stub("moe_wna16_marlin_gemm")
prepare_moe_input = _stub("prepare_moe_input")
topk_sigmoid = _stub("topk_sigmoid")
topk_softmax = _stub("topk_softmax")

# ── expert specialization ─────────────────────────────────────────────────────
es_fp8_blockwise_scaled_grouped_mm = _stub("es_fp8_blockwise_scaled_grouped_mm")
es_sm100_mxfp8_blockscaled_grouped_mm = _stub("es_sm100_mxfp8_blockscaled_grouped_mm")
es_sm100_mxfp8_blockscaled_grouped_quant = _stub("es_sm100_mxfp8_blockscaled_grouped_quant")

# ── quantization ─────────────────────────────────────────────────────────────
ggml_dequantize = _stub("ggml_dequantize")
ggml_moe_a8 = _stub("ggml_moe_a8")
ggml_moe_a8_vec = _stub("ggml_moe_a8_vec")
ggml_moe_get_block_size = _stub("ggml_moe_get_block_size")
ggml_mul_mat_a8 = _stub("ggml_mul_mat_a8")
ggml_mul_mat_vec_a8 = _stub("ggml_mul_mat_vec_a8")

# ── marlin ────────────────────────────────────────────────────────────────────
awq_marlin_moe_repack = _stub("awq_marlin_moe_repack")
awq_marlin_repack = _stub("awq_marlin_repack")
gptq_marlin_repack = _stub("gptq_marlin_repack")

# ── memory ────────────────────────────────────────────────────────────────────
set_kv_buffer_kernel = _stub("set_kv_buffer_kernel")
weak_ref_tensor = _stub("weak_ref_tensor")

# ── sampling — PyTorch fallbacks ──────────────────────────────────────────────

def top_k_renorm_prob(probs: "_torch.Tensor", top_ks: "_torch.Tensor") -> "_torch.Tensor":
    """Zero out below-top-k entries per row, renormalize."""
    B, V = probs.shape
    out = probs.clone()
    ks = top_ks.clamp(1, V).tolist()
    for i, k in enumerate(ks):
        if k < V:
            kth = float(probs[i].topk(k).values[-1])
            out[i] = _torch.where(probs[i] >= kth, probs[i], _torch.zeros_like(probs[i]))
    sums = out.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return out / sums


def top_p_renorm_prob(probs: "_torch.Tensor", top_ps: "_torch.Tensor") -> "_torch.Tensor":
    """Zero out tail entries below nucleus threshold, renormalize."""
    sorted_probs, sorted_idx = _torch.sort(probs, dim=-1, descending=True)
    cum = _torch.cumsum(sorted_probs, dim=-1)
    # shift by 1 so the token that crosses p is kept
    mask = (cum - sorted_probs) >= top_ps.unsqueeze(-1)
    sorted_probs[mask] = 0.0
    # scatter back to original order
    out = _torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    sums = out.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return out / sums


def top_k_top_p_sampling_from_probs(
    probs: "_torch.Tensor",
    top_ks: "_torch.Tensor",
    top_ps: "_torch.Tensor",
    filter_apply_order: str = "joint",
    check_nan: bool = False,
) -> "_torch.Tensor":
    """Top-k + top-p filtering then multinomial sampling."""
    if check_nan and _torch.isnan(probs).any():
        probs = _torch.where(_torch.isnan(probs), _torch.zeros_like(probs), probs)
    filtered = top_k_renorm_prob(probs, top_ks)
    filtered = top_p_renorm_prob(filtered, top_ps)
    return _torch.multinomial(filtered, num_samples=1).squeeze(-1)


def top_p_sampling_from_probs(
    probs: "_torch.Tensor", top_ps: "_torch.Tensor", **kwargs
) -> "_torch.Tensor":
    filtered = top_p_renorm_prob(probs, top_ps)
    return _torch.multinomial(filtered, num_samples=1).squeeze(-1)


def min_p_sampling_from_probs(
    probs: "_torch.Tensor", min_ps: "_torch.Tensor", **kwargs
) -> "_torch.Tensor":
    top_probs = probs.max(dim=-1, keepdim=True).values
    threshold = min_ps.unsqueeze(-1) * top_probs
    filtered = _torch.where(probs >= threshold, probs, _torch.zeros_like(probs))
    sums = filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    filtered = filtered / sums
    return _torch.multinomial(filtered, num_samples=1).squeeze(-1)


def top_k_top_p_sampling_from_logits(
    logits: "_torch.Tensor",
    top_ks: "_torch.Tensor",
    top_ps: "_torch.Tensor",
    **kwargs,
) -> "_torch.Tensor":
    probs = _torch.softmax(logits.float(), dim=-1)
    return top_k_top_p_sampling_from_probs(probs, top_ks, top_ps, **kwargs)


def top_k_mask_logits(
    logits: "_torch.Tensor", top_ks: "_torch.Tensor"
) -> "_torch.Tensor":
    B, V = logits.shape
    out = logits.clone()
    ks = top_ks.clamp(1, V).tolist()
    for i, k in enumerate(ks):
        if k < V:
            kth = float(logits[i].topk(k).values[-1])
            out[i] = _torch.where(logits[i] >= kth, logits[i],
                                  _torch.full_like(logits[i], float("-inf")))
    return out

# ── speculative ───────────────────────────────────────────────────────────────
build_tree_kernel_efficient = _stub("build_tree_kernel_efficient")
reconstruct_indices_from_tree_mask = _stub("reconstruct_indices_from_tree_mask")
segment_packbits = _stub("segment_packbits")
tree_speculative_sampling_target_only = _stub("tree_speculative_sampling_target_only")
verify_tree_greedy = _stub("verify_tree_greedy")

# ── top_k ─────────────────────────────────────────────────────────────────────
fast_topk = _stub("fast_topk")
fast_topk_transform_fused = _stub("fast_topk_transform_fused")
fast_topk_transform_ragged_fused = _stub("fast_topk_transform_ragged_fused")
fast_topk_v2 = _stub("fast_topk_v2")

# ── grammar ───────────────────────────────────────────────────────────────────
apply_token_bitmask_inplace_cuda = _stub("apply_token_bitmask_inplace_cuda")

# ── hadamard ──────────────────────────────────────────────────────────────────
hadamard_transform = _stub("hadamard_transform")
hadamard_transform_12n = _stub("hadamard_transform_12n")
hadamard_transform_20n = _stub("hadamard_transform_20n")
hadamard_transform_28n = _stub("hadamard_transform_28n")
hadamard_transform_40n = _stub("hadamard_transform_40n")

# ── mamba ─────────────────────────────────────────────────────────────────────
causal_conv1d_fwd = _stub("causal_conv1d_fwd")
causal_conv1d_update = _stub("causal_conv1d_update")

# ── spatial (lazy) ────────────────────────────────────────────────────────────
def create_greenctx_stream_by_value(*a, **kw):
    raise NotImplementedError("sgl_kernel.create_greenctx_stream_by_value: stub")

def get_sm_available(*a, **kw):
    raise NotImplementedError("sgl_kernel.get_sm_available: stub")

# ── allreduce (stub module) ───────────────────────────────────────────────────
# Must be importable as sgl_kernel.allreduce (attribute access, not just submodule).
from sgl_kernel import allreduce  # noqa: E402  makes sgl_kernel.allreduce work
