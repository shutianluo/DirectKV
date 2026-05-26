/******************************************************************************
 * Host-side launch helpers for Projection-Fused Flash Attention (Phase 1).
 ******************************************************************************/
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.cuh"

namespace proj_fused {

static CUtensorMap* g_tma_desc_Q_dev = nullptr;

inline CUresult create_tma_desc_Q_4d(
    CUtensorMap* desc, void const* ptr_Q,
    int batch, int seqlen, int num_heads, int head_dim)
{
    // Q: [B, S, NH, D] row-major → 4D TMA: (D, NH, S, B)
    uint64_t gDim[4] = {(uint64_t)head_dim, (uint64_t)num_heads,
                         (uint64_t)seqlen, (uint64_t)batch};
    uint64_t gStr[3] = {
        (uint64_t)head_dim * 2,                           // NH stride (D * sizeof(fp16))
        (uint64_t)num_heads * head_dim * 2,               // S stride
        (uint64_t)seqlen * num_heads * head_dim * 2       // B stride
    };
    uint32_t box[4] = {64, 1, 64, 1};
    uint32_t eStr[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4,
        const_cast<void*>(ptr_Q), gDim, gStr, box, eStr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <typename Traits>
cudaError_t run_proj_fused_fwd(
    void const* ptr_X, void const* ptr_Wk, void const* ptr_Wv, void const* ptr_Q,
    void* ptr_Opart, void* ptr_Kc, void* ptr_Vc, float* ptr_LSE,
    int batch, int seqlen, int num_heads, int hidden_dim, int head_dim,
    float softmax_scale, bool is_causal,
    cudaStream_t stream,
    void const* ptr_Kpast = nullptr, void const* ptr_Vpast = nullptr, int seqlen_past = 0)
{
    if (seqlen_past > 0 && (seqlen_past % Traits::kBlockN) != 0) return cudaErrorInvalidValue;

    ProjFusedParams p;
    p.ptr_X = ptr_X;  p.ptr_Wk = ptr_Wk;  p.ptr_Wv = ptr_Wv;  p.ptr_Q = ptr_Q;
    p.ptr_Opart = ptr_Opart;  p.ptr_Kc = ptr_Kc;  p.ptr_Vc = ptr_Vc;  p.ptr_LSE = ptr_LSE;
    p.ptr_Kpast = ptr_Kpast;  p.ptr_Vpast = ptr_Vpast;
    p.batch = batch;  p.seqlen = seqlen;  p.num_heads = num_heads;
    p.hidden_dim = hidden_dim;  p.head_dim = head_dim;
    p.seqlen_past = seqlen_past;
    p.num_n_blocks_past = seqlen_past / Traits::kBlockN;
    p.num_n_blocks = p.num_n_blocks_past + (seqlen + Traits::kBlockN - 1) / Traits::kBlockN;
    p.num_m_blocks = (seqlen + Traits::kBlockM - 1) / Traits::kBlockM;
    p.softmax_scale = softmax_scale;
    p.is_causal = is_causal ? 1 : 0;

    // Create 4D TMA descriptor for Q (persistent device allocation).
    if (!g_tma_desc_Q_dev) {
        cudaError_t ae = cudaMalloc(&g_tma_desc_Q_dev, sizeof(CUtensorMap));
        if (ae != cudaSuccess) return ae;
    }
    {
        CUtensorMap h;
        CUresult cr = create_tma_desc_Q_4d(&h, ptr_Q, batch, seqlen, num_heads, head_dim);
        if (cr != CUDA_SUCCESS) return cudaErrorInvalidValue;
        cudaError_t ce = cudaMemcpyAsync(g_tma_desc_Q_dev, &h, sizeof(CUtensorMap),
                                          cudaMemcpyHostToDevice, stream);
        if (ce != cudaSuccess) return ce;
    }
    p.tma_desc_Q = g_tma_desc_Q_dev;

    dim3 grid(p.num_n_blocks, num_heads, batch);
    dim3 block(Traits::kNThreadsMMA);  // 128 threads only (no storer warp)

    auto round128 = [](size_t s) { return (s + 127) & ~size_t(127); };
    // Aliased smem layout: sX⊂sQ0, sWk=sO0, sWv=sO1.
    // Layout: sQ0(16K) | sQ1(16K) | sK(16K) | sV(16K) | sO0/sWk(16K) | sO1/sWv(16K)
    size_t smem_sz = 128;
    smem_sz += round128(Traits::kSmemBytesQ);   // sQ0 (also hosts sX alias)
    smem_sz += round128(Traits::kSmemBytesQ);   // sQ1
    smem_sz += round128(Traits::kSmemBytesK);
    smem_sz += round128(Traits::kSmemBytesV);
    smem_sz += round128(Traits::kSmemBytesO);   // sO0 / sWk (aliased)
    smem_sz += round128(Traits::kSmemBytesO);   // sO1 / sWv (aliased)
    smem_sz += 128;  // mbarrier_Q[2] + alignment
    auto* kernel = &proj_fused_fwd_kernel<Traits>;
    cudaError_t err = cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_sz);
    if (err != cudaSuccess) return err;
    kernel<<<grid, block, smem_sz, stream>>>(p);
    return cudaGetLastError();
}

template <typename Element, int kHeadDim>
cudaError_t run_combine(
    void const* ptr_Opart, float const* ptr_LSE, void* ptr_O,
    int batch, int seqlen, int num_heads, int head_dim, int num_n_blocks,
    cudaStream_t stream)
{
    CombineParams p;
    p.ptr_Opart = ptr_Opart;  p.ptr_LSE = ptr_LSE;  p.ptr_O = ptr_O;
    p.batch = batch;  p.seqlen = seqlen;  p.num_heads = num_heads;
    p.head_dim = head_dim;  p.num_n_blocks = num_n_blocks;
    dim3 grid(seqlen, num_heads, batch);
    dim3 block(kHeadDim);
    constexpr int kTileN = 64;
    int num_n_padded = (num_n_blocks + 3) & ~3;  // round up to 16-byte alignment
    size_t smem_combine = (size_t)num_n_padded * sizeof(float)
                        + (size_t)kTileN * kHeadDim * sizeof(Element);
    auto* ck = &proj_fused_combine_kernel<Element, kHeadDim>;
    cudaError_t err2 = cudaFuncSetAttribute(ck,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_combine);
    if (err2 != cudaSuccess) return err2;
    ck<<<grid, block, smem_combine, stream>>>(p);
    return cudaGetLastError();
}

} // namespace proj_fused
