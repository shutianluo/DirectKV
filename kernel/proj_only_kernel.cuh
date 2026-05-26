/******************************************************************************
 * Phase 1 — Projection-Only Kernel (SM90)
 *
 * Computes K = X @ Wk^T and V = X @ Wv^T for one KV tile per CTA,
 * then writes K and V to global memory in [B, S, NH, D] row-major layout.
 *
 * Thread organization (256 threads / 2 warpgroups):
 *   WG0 (tid 0-127):   cp.async.cg loads for X, Wk, Wv (Phase 1 only).
 *   WG1 (tid 128-255): cp.async.cg loads + WGMMA for K/V projection.
 *
 * No TMA, no attention, no aliasing — correctness vehicle only.
 *
 * Correctness gates: K max_abs_err < 5e-3,  V max_abs_err < 5e-3
 *   against X @ Wk^T and X @ Wv^T.
 ******************************************************************************/
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/array.h>

#include "proj_fused_kernel_traits_sm90.h"
#include "utils.h"

namespace proj_only {

using namespace cute;

// ---------------------------------------------------------------------------
// Param struct — raw pointers only, no TMA descriptors
// ---------------------------------------------------------------------------
struct ProjOnlyParams {
    void const* __restrict__ ptr_X;   // [B, S, H]         input token embeddings
    void const* __restrict__ ptr_Wk;  // [NH, D, H]        key projection weight
    void const* __restrict__ ptr_Wv;  // [NH, D, H]        value projection weight
    void*       __restrict__ ptr_Kc;  // [B, S, NH, D]     output K (written)
    void*       __restrict__ ptr_Vc;  // [B, S, NH, D]     output V (written)
    int batch, seqlen, num_heads, hidden_dim, head_dim;
    // num_n_blocks = ceil(seqlen / kBlockN), set by launch wrapper
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <typename Traits>
__global__ __launch_bounds__(256, 1)
void proj_only_fwd_kernel(ProjOnlyParams params)
{
    using Element = typename Traits::Element;
    constexpr int kBlockN       = Traits::kBlockN;
    constexpr int kHeadDim      = Traits::kHeadDim;
    constexpr int kHiddenChunk  = Traits::kHiddenChunk;
    constexpr int kNumProjChunks = Traits::kNumProjChunks;
    using SmemLayoutK  = typename Traits::SmemLayoutK;
    using SmemLayoutV  = typename Traits::SmemLayoutV;
    using SmemLayoutX  = typename Traits::SmemLayoutX;
    using SmemLayoutW  = typename Traits::SmemLayoutW;
    using TiledMmaProj = typename Traits::TiledMmaProj;

    const int n_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int tid     = threadIdx.x;

    const int wg_idx = tid / 128;       // 0 = WG0, 1 = WG1 (compute)
    const int ctid   = tid - 128;       // WG1-local tid [0,127]; valid only for wg_idx==1

    const int S  = params.seqlen;
    const int H  = params.hidden_dim;
    const int NH = params.num_heads;
    const int n_start = n_block * kBlockN;

    auto const* X_ptr  = reinterpret_cast<Element const*>(params.ptr_X);
    auto const* Wk_ptr = reinterpret_cast<Element const*>(params.ptr_Wk);
    auto const* Wv_ptr = reinterpret_cast<Element const*>(params.ptr_Wv);
    auto* Kc_ptr = reinterpret_cast<Element*>(params.ptr_Kc);
    auto* Vc_ptr = reinterpret_cast<Element*>(params.ptr_Vc);

    // Base pointers for this CTA's tile
    Element const* gX  = X_ptr  + (size_t)batch * S * H + (size_t)n_start * H;
    Element const* gWk = Wk_ptr + (size_t)head * kHeadDim * H;
    Element const* gWv = Wv_ptr + (size_t)head * kHeadDim * H;

    // ------------------------------------------------------------------
    // Shared memory — non-aliased, five separate buffers
    //   sX  [kBlockN, kHiddenChunk]    8 KB
    //   sWk [kHeadDim, kHiddenChunk]  16 KB
    //   sWv [kHeadDim, kHiddenChunk]  16 KB
    //   sK  [kBlockN, kHeadDim]       16 KB
    //   sV  [kHeadDim, kBlockN]       16 KB   (MN-major)
    //   Total ~72 KB
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto align128 = [](uintptr_t p) -> uintptr_t { return (p + 127) & ~uintptr_t(127); };
    uintptr_t base = align128(reinterpret_cast<uintptr_t>(smem_raw));

    Element* sX_ptr  = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutX{}) * sizeof(Element));
    Element* sWk_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutW{}) * sizeof(Element));
    Element* sWv_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutW{}) * sizeof(Element));
    Element* sK_ptr  = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutK{}) * sizeof(Element));
    Element* sV_ptr  = reinterpret_cast<Element*>(base);

    Tensor sX  = make_tensor(make_smem_ptr(sX_ptr),  SmemLayoutX{});
    Tensor sWk = make_tensor(make_smem_ptr(sWk_ptr), SmemLayoutW{});
    Tensor sWv = make_tensor(make_smem_ptr(sWv_ptr), SmemLayoutW{});
    Tensor sK  = make_tensor(make_smem_ptr(sK_ptr),  SmemLayoutK{});
    Tensor sV  = make_tensor(make_smem_ptr(sV_ptr),  SmemLayoutV{});

    // ------------------------------------------------------------------
    // Tiled MMA + accumulator fragments (WG1 only, but declared for all;
    // WG0 skips WGMMA instructions).
    // ------------------------------------------------------------------
    TiledMmaProj tiled_mma_proj;
    auto thr_mma = tiled_mma_proj.get_thread_slice(wg_idx == 1 ? ctid : 0);

    Tensor acc_k = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_v = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    if (wg_idx == 1) { clear(acc_k); clear(acc_v); }

    Tensor tCsX  = thr_mma.partition_fragment_A(sX);
    Tensor tCsWk = thr_mma.partition_fragment_B(sWk);
    Tensor tCsWv = thr_mma.partition_fragment_B(sWv);

    // ------------------------------------------------------------------
    // Load helpers (all 256 threads issue cp.async.cg)
    // ------------------------------------------------------------------
    constexpr int kElemsPerVec = 8;   // 16 bytes / sizeof(fp16) = 8 elements per cp.async.cg

    // Load a chunk of X: shape [kBlockN, kHiddenChunk], row-major in global
    auto load_X_all = [&](int cs) {
        constexpr int kVPR = kHiddenChunk / kElemsPerVec;
        constexpr int kTV  = kBlockN * kVPR;
        constexpr int kVPT = (kTV + 255) / 256;
        #pragma unroll
        for (int v = 0; v < kVPT; ++v) {
            int idx = v * 256 + tid;
            if (idx < kTV) {
                int row = idx / kVPR;
                int col = (idx % kVPR) * kElemsPerVec;
                uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(&sX(row, col)));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sa), "l"(gX + (size_t)row * H + cs * kHiddenChunk + col));
            }
        }
    };

    // Load a chunk of a weight matrix: shape [kHeadDim, kHiddenChunk], row-major in global
    // gSrc[row, col] = gSrc + row * H + cs * kHiddenChunk + col
    auto load_W_all = [&](Element const* gSrc, int cs, Element* sDst_ptr, SmemLayoutW) {
        Tensor sDst = make_tensor(make_smem_ptr(sDst_ptr), SmemLayoutW{});
        constexpr int kVPR = kHiddenChunk / kElemsPerVec;
        constexpr int kTV  = kHeadDim * kVPR;
        constexpr int kVPT = (kTV + 255) / 256;
        #pragma unroll
        for (int v = 0; v < kVPT; ++v) {
            int idx = v * 256 + tid;
            if (idx < kTV) {
                int row = idx / kVPR;
                int col = (idx % kVPR) * kElemsPerVec;
                uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(&sDst(row, col)));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sa), "l"(gSrc + (size_t)row * H + cs * kHiddenChunk + col));
            }
        }
    };

    // ------------------------------------------------------------------
    // Inner projection loop — synchronous cp.async pattern
    // ------------------------------------------------------------------
    #pragma unroll 1
    for (int cs = 0; cs < kNumProjChunks; ++cs) {
        load_X_all(cs);
        load_W_all(gWk, cs, sWk_ptr, SmemLayoutW{});
        load_W_all(gWv, cs, sWv_ptr, SmemLayoutW{});
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        if (wg_idx == 1) {
            if (cs == 0) {
                flash::gemm<true,  -1>(tiled_mma_proj, tCsX, tCsWk, acc_k);
                flash::gemm<true,  -1>(tiled_mma_proj, tCsX, tCsWv, acc_v);
            } else {
                flash::gemm<false, -1>(tiled_mma_proj, tCsX, tCsWk, acc_k);
                flash::gemm<false, -1>(tiled_mma_proj, tCsX, tCsWv, acc_v);
            }
            cute::warpgroup_wait<0>();
            cute::warpgroup_fence_operand(acc_k);
            cute::warpgroup_fence_operand(acc_v);
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------
    // WG1 only: write accumulator results into sK and sV
    //   sK: straightforward, partition_C gives correct placement
    //   sV: transpose (i, d) → sV(d, i) using identity coordinate tensor
    // ------------------------------------------------------------------
    if (wg_idx == 1) {
        Tensor tCsK  = thr_mma.partition_C(sK);
        // Identity coordinate tensor for V: maps each acc element to (row, col) in [kBlockN, kHeadDim]
        Tensor cV    = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tCcV  = thr_mma.partition_C(cV);
        #pragma unroll
        for (int i = 0; i < size(acc_k); ++i) {
            tCsK(i) = Element(acc_k(i));
            // Transpose: logical V[row=get<0>, col=get<1>] → sV[col, row] (MN-major)
            sV(get<1>(tCcV(i)), get<0>(tCcV(i))) = Element(acc_v(i));
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // All 256 threads: store sK → Kc and sV → Vc (uint4, row-major)
    //   Kc layout: [B, S, NH, D]  — sK(i, d) → Kc[batch, n_start+i, head, d]
    //   Vc layout: [B, S, NH, D]  — sV(d, i) = V[i][d] → Vc[batch, n_start+i, head, d]
    // ------------------------------------------------------------------
    constexpr int kEPV   = 8;                          // elements per uint4 vector
    constexpr int kVPR   = kHeadDim / kEPV;            // vectors per row (head dim)
    constexpr int kTV    = kBlockN * kVPR;             // total vectors per tile
    constexpr int kVPT   = (kTV + 255) / 256;          // vectors per thread

    #pragma unroll
    for (int v = 0; v < kVPT; ++v) {
        int idx = v * 256 + tid;
        if (idx < kTV) {
            int i = idx / kVPR;                        // row within KV tile [0, kBlockN)
            int d = (idx % kVPR) * kEPV;               // column base [0, kHeadDim) in steps of 8
            int row = n_start + i;
            if (row < S) {
                size_t off = ((size_t)batch * S + row) * NH * kHeadDim
                           + (size_t)head * kHeadDim + d;
                // Pack K from sK(i, d..d+7)
                uint4 kv_k;
                #pragma unroll
                for (int k = 0; k < kEPV; ++k)
                    reinterpret_cast<Element*>(&kv_k)[k] = sK(i, d + k);
                *reinterpret_cast<uint4*>(Kc_ptr + off) = kv_k;

                // Pack V from sV(d..d+7, i)  [MN-major: first index is head-dim]
                uint4 kv_v;
                #pragma unroll
                for (int k = 0; k < kEPV; ++k)
                    reinterpret_cast<Element*>(&kv_v)[k] = sV(d + k, i);
                *reinterpret_cast<uint4*>(Vc_ptr + off) = kv_v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------

template <typename Traits>
cudaError_t launch_proj_only(
    void const* ptr_X, void const* ptr_Wk, void const* ptr_Wv,
    void* ptr_Kc, void* ptr_Vc,
    int batch, int seqlen, int num_heads, int hidden_dim,
    cudaStream_t stream = 0)
{
    ProjOnlyParams p;
    p.ptr_X = ptr_X; p.ptr_Wk = ptr_Wk; p.ptr_Wv = ptr_Wv;
    p.ptr_Kc = ptr_Kc; p.ptr_Vc = ptr_Vc;
    p.batch = batch; p.seqlen = seqlen;
    p.num_heads = num_heads; p.hidden_dim = hidden_dim;
    p.head_dim = Traits::kHeadDim;

    int num_n_blocks = (seqlen + Traits::kBlockN - 1) / Traits::kBlockN;
    dim3 grid(num_n_blocks, num_heads, batch);
    dim3 block(256);

    auto r128 = [](size_t s) { return (s + 127) & ~size_t(127); };
    size_t smem = 128;
    smem += r128(cute::cosize(typename Traits::SmemLayoutX{}) * sizeof(typename Traits::Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutW{}) * sizeof(typename Traits::Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutW{}) * sizeof(typename Traits::Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutK{}) * sizeof(typename Traits::Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutV{}) * sizeof(typename Traits::Element));

    auto* kernel = &proj_only_fwd_kernel<Traits>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (err != cudaSuccess) return err;
    kernel<<<grid, block, smem, stream>>>(p);
    return cudaGetLastError();
}

} // namespace proj_only
