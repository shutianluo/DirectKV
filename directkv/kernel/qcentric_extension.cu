/*
 * DirectKV CUDA Extension
 * ========================
 * Python-callable wrapper around the qcentric_attn_kernel from
 * flash-attention/autoresearch/kernel/kernel_qcentric.cuh.
 *
 * This wrapper:
 *   1. Takes Q (GPU), Kpast/Vpast (CPU-pinned), Knew/Vnew (CPU-pinned or GPU).
 *   2. Allocates a temporary Opart buffer.
 *   3. Launches qcentric_attn_kernel (Pass 2 only — projection is not needed
 *      because SGLang has already computed K/V).
 *   4. Returns O = Opart[:,:,:,0,:] (single n-block, so no combine needed).
 *
 * Compile requirements:
 *   - CUDA >= 12.0, SM90 (Hopper)
 *   - CUTLASS >= 3.3 headers (for CuTe / WGMMA descriptors)
 *   - -arch=sm_90 -std=c++17 --expt-relaxed-constexpr
 *   - -DCUTE_ARCH_MMA_SM90A_ENABLED -DCUTLASS_ARCH_MMA_SM90_ENABLED
 *
 * Build via the JIT loader in __init__.py or the AOT setup.py.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// Include the two-pass Q-centric kernel
#include "kernel_qcentric.cuh"

#define CHECK_CUDA(x)  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x)   TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

/* --------------------------------------------------------------------------
 * Helper: pick DefaultTraits based on head_dim and element dtype.
 * For the MVP we support fp16, head_dim=128. BF16 shares the same layout.
 * -------------------------------------------------------------------------- */
using DefaultFP16Traits = proj_fused::ProjFusedKernelTraitsSm90<
    64, 64, 128, 128, 64, cutlass::half_t>;

/* --------------------------------------------------------------------------
 * directkv_fwd_impl<Traits>
 *
 * Launches qcentric_attn_kernel for a single (B, S_new=1) decode step or a
 * full (B, S_new>1) extend step.  K/V past tensors may live in CPU-pinned
 * memory; the kernel reads them via cp.async.cg over PCIe/NVLink.
 *
 * Tensor layout expected (all innermost dim = head_dim, contiguous):
 *   Q     : [B, S_new, NH_q,  D]   GPU
 *   Kpast : [B, S_past, NH_kv, D]  CPU-pinned, S_past % 64 == 0
 *   Vpast : [B, S_past, NH_kv, D]  CPU-pinned
 *   Knew  : [B, S_new,  NH_kv, D]  CPU-pinned (already stored by set_kv_buffer)
 *   Vnew  : [B, S_new,  NH_kv, D]  CPU-pinned
 *   O     : [B, S_new, NH_q,  D]   GPU output (allocated here)
 * -------------------------------------------------------------------------- */
template <typename Traits>
torch::Tensor directkv_fwd_impl(
    torch::Tensor Q,       // GPU
    torch::Tensor Kpast,   // CPU-pinned
    torch::Tensor Vpast,   // CPU-pinned
    torch::Tensor Knew,    // CPU-pinned
    torch::Tensor Vnew,    // CPU-pinned
    float softmax_scale,
    bool  is_causal)
{
    using Element = typename Traits::Element;

    const int B      = Q.size(0);
    const int S_new  = Q.size(1);
    const int NH_q   = Q.size(2);
    const int D      = Q.size(3);
    const int NH_kv  = Kpast.size(2);
    const int S_past = Kpast.size(1);

    // num_heads seen by kernel = NH_q (the kernel tiles over heads)
    // NOTE: qcentric_attn_kernel supports GQA only if NH_kv == NH_q.
    //       For GQA (NH_q > NH_kv) we expand K/V before the call.
    //       This expansion is handled on the Python side in _forward_decode_kernel
    //       (both K/V are already repeated to NH_q).
    TORCH_CHECK(NH_q == NH_kv,
        "directkv_extension: NH_q (", NH_q, ") != NH_kv (", NH_kv, "). "
        "Expand K/V heads to NH_q before calling directkv_fwd.");
    TORCH_CHECK(S_past % Traits::kBlockN == 0,
        "S_past (", S_past, ") must be a multiple of kBlockN=", Traits::kBlockN,
        ". Pad the past sequence before calling.");

    constexpr int kBM = Traits::kBlockM;
    constexpr int kBN = Traits::kBlockN;

    const int num_n_past   = S_past / kBN;
    const int num_n_new    = (S_new + kBN - 1) / kBN;
    const int num_n_blocks = num_n_past + num_n_new;
    const int num_m_blocks = (S_new + kBM - 1) / kBM;

    // Opart: [B, S_new, NH, num_n_alloc, D]  with num_n_alloc=1
    // (the kernel writes n=0 and sets LSE[n>0]=-INF; with 1 block the
    //  final output equals Opart directly without a combine pass)
    const int num_n_alloc = 1;
    auto O = torch::zeros({B, S_new, NH_q, D}, Q.options());  // GPU
    auto Opart = O;  // Reuse output buffer; shape broadcast OK for n=1 case
    // Actually allocate separate Opart with 5D to match kernel's write pattern
    auto Opart5 = torch::zeros({B, S_new, NH_q, num_n_alloc, D}, Q.options());
    auto LSE    = torch::full({B, S_new, NH_q, num_n_alloc}, -1e9f,
                               Q.options().dtype(torch::kFloat32));

    proj_fused::TwoPassParams params;
    params.ptr_Q     = Q.data_ptr();
    params.ptr_Kpast = Kpast.data_ptr();
    params.ptr_Vpast = Vpast.data_ptr();
    params.ptr_Kc    = Knew.data_ptr();
    params.ptr_Vc    = Vnew.data_ptr();
    params.ptr_Opart = Opart5.data_ptr();
    params.ptr_LSE   = LSE.data_ptr<float>();
    params.ptr_Wk    = nullptr;  // not used in Pass 2
    params.ptr_Wv    = nullptr;
    params.ptr_X     = nullptr;

    params.batch         = B;
    params.seqlen        = S_new;
    params.num_heads     = NH_q;
    params.hidden_dim    = D;    // unused by qcentric_attn_kernel
    params.head_dim      = D;
    params.seqlen_past       = S_past;
    params.num_n_blocks_past = num_n_past;
    params.num_n_blocks      = num_n_blocks;
    params.num_m_blocks      = num_m_blocks;
    params.softmax_scale     = softmax_scale;
    params.is_causal         = is_causal ? 1 : 0;

    // Shared memory for qcentric_attn_kernel: sQ + sK + sV
    auto r128 = [](size_t s) -> size_t { return (s + 127) & ~size_t(127); };
    size_t smem = r128(Traits::kSmemBytesQ)
                + r128(Traits::kSmemBytesK)
                + r128(Traits::kSmemBytesV)
                + 256;  // alignment padding

    auto* kernel_fn = &proj_fused::qcentric_attn_kernel<Traits>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    TORCH_CHECK(err == cudaSuccess,
        "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(num_m_blocks, NH_q, B);
    dim3 block(Traits::kNThreadsMMA);
    kernel_fn<<<grid, block, smem, stream>>>(params);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "qcentric_attn_kernel launch failed: ", cudaGetErrorString(err));

    // Extract Opart[:,:,:,0,:] → O
    // Opart5: [B, S_new, NH, 1, D] → squeeze dim 3
    return Opart5.squeeze(3);  // [B, S_new, NH, D]
}

/* --------------------------------------------------------------------------
 * Python-visible entry point
 * -------------------------------------------------------------------------- */
torch::Tensor directkv_fwd(
    torch::Tensor Q,
    torch::Tensor Kpast,
    torch::Tensor Vpast,
    torch::Tensor Knew,
    torch::Tensor Vnew,
    double softmax_scale,
    bool   is_causal)
{
    // Validate shapes
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, S_new, NH_q, D]");
    TORCH_CHECK(Kpast.dim() == 4, "Kpast must be 4D [B, S_past, NH_kv, D]");
    TORCH_CHECK(Vpast.dim() == 4, "Vpast must be 4D [B, S_past, NH_kv, Dv]");
    TORCH_CHECK(Knew.dim() == 4, "Knew must be 4D [B, S_new, NH_kv, D]");
    TORCH_CHECK(Vnew.dim() == 4, "Vnew must be 4D [B, S_new, NH_kv, Dv]");

    CHECK_CUDA(Q);
    // Kpast/Vpast/Knew/Vnew may be CPU-pinned; don't CHECK_CUDA on them.
    CHECK_CONTIG(Q);
    CHECK_CONTIG(Kpast);
    CHECK_CONTIG(Vpast);
    CHECK_CONTIG(Knew);
    CHECK_CONTIG(Vnew);

    const int D = Q.size(3);
    TORCH_CHECK(D == 128,
        "directkv_extension: head_dim must be 128 (DefaultFP16Traits). "
        "Got D=", D, ".");

    auto dtype = Q.scalar_type();
    TORCH_CHECK(
        dtype == at::kHalf || dtype == at::kBFloat16,
        "directkv_extension: only float16 / bfloat16 supported, got ", dtype);

    // For BF16, cast to FP16 for the kernel (DefaultFP16Traits uses __half).
    // This is an MVP limitation; a BF16 specialisation can be added later.
    bool need_cast = (dtype == at::kBFloat16);
    if (need_cast) {
        Q     = Q.to(at::kHalf);
        Kpast = Kpast.to(at::kHalf);
        Vpast = Vpast.to(at::kHalf);
        Knew  = Knew.to(at::kHalf);
        Vnew  = Vnew.to(at::kHalf);
    }

    auto O = directkv_fwd_impl<DefaultFP16Traits>(
        Q, Kpast, Vpast, Knew, Vnew, (float)softmax_scale, is_causal);

    if (need_cast) {
        O = O.to(dtype);
    }
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DirectKV: attention with CPU-pinned KV cache.";
    m.def("directkv_fwd", &directkv_fwd,
          "DirectKV attention forward pass.\n\n"
          "Args:\n"
          "  Q     (Tensor): [B, S_new, NH_q, D]  GPU, fp16 or bf16\n"
          "  Kpast (Tensor): [B, S_past, NH_kv, D] CPU-pinned, S_past%64==0\n"
          "  Vpast (Tensor): [B, S_past, NH_kv, D] CPU-pinned\n"
          "  Knew  (Tensor): [B, S_new, NH_kv, D]  CPU-pinned\n"
          "  Vnew  (Tensor): [B, S_new, NH_kv, D]  CPU-pinned\n"
          "  softmax_scale (float): attention scale (1/sqrt(D))\n"
          "  is_causal (bool): apply causal mask\n"
          "Returns:\n"
          "  O (Tensor): [B, S_new, NH_q, D] GPU");
}
