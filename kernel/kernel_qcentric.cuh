/******************************************************************************
 * Two-pass Projection-Fused Flash Attention (SM90)
 *
 * Pass 1: Project X -> K,V (KV-centric, 1 CTA per KV block × head)
 * Pass 2: Q-centric attention reading K/V from HBM (no partial outputs)
 *
 * Eliminates the Opart intermediate and combine kernel entirely.
 ******************************************************************************/
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/array.h>

#include "proj_fused_kernel_traits_sm90.h"
#include "softmax.h"
#include "utils.h"

namespace proj_fused {

using namespace cute;

// Shared params structure
struct TwoPassParams {
    void const* __restrict__ ptr_X;   // [B, S_new, H]
    void const* __restrict__ ptr_Wk;  // [num_heads, D, H]
    void const* __restrict__ ptr_Wv;  // [num_heads, D, H]
    void const* __restrict__ ptr_Q;   // [B, S_new, num_heads, D]
    void*       __restrict__ ptr_Opart;  // [B, S, NH, num_n_blocks, D] — write to n=0
    void*       __restrict__ ptr_Kc;  // [B, S_new, num_heads, D]
    void*       __restrict__ ptr_Vc;  // [B, S_new, num_heads, D]
    float*      __restrict__ ptr_LSE; // [B, S, NH, num_n_blocks]
    void const* __restrict__ ptr_Kpast;
    void const* __restrict__ ptr_Vpast;
    int batch, seqlen, num_heads, hidden_dim, head_dim;
    int seqlen_past;
    int num_n_blocks_past;
    int num_n_blocks;
    int num_m_blocks;
    float softmax_scale;
    int is_causal;
};

////////////////////////////////////////////////////////////////////////////////
// Pass 1: Projection kernel — projects X -> K, V, stores to Kc/Vc
// Grid: (num_n_blocks_new, num_heads, batch)
////////////////////////////////////////////////////////////////////////////////
template <typename Traits>
__global__ __launch_bounds__(Traits::kNThreadsMMA, 2)
void proj_kernel(TwoPassParams params)
{
    using Element = typename Traits::Element;
    constexpr int kBlockN   = Traits::kBlockN;
    constexpr int kHeadDim  = Traits::kHeadDim;
    constexpr int kHiddenChunk = Traits::kHiddenChunk;
    constexpr int kNumProjChunks = Traits::kNumProjChunks;
    constexpr int kNThreadsMMA = Traits::kNThreadsMMA;
    using SmemLayoutK = typename Traits::SmemLayoutK;
    using SmemLayoutV = typename Traits::SmemLayoutV;
    using SmemLayoutX = typename Traits::SmemLayoutX;
    using SmemLayoutW = typename Traits::SmemLayoutW;
    using TiledMmaProj = typename Traits::TiledMmaProj;

    const int n_block_new = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int tid     = threadIdx.x;

    const int S = params.seqlen;
    const int H = params.hidden_dim;
    const int D = params.head_dim;
    const int NH = params.num_heads;

    auto const* X_ptr  = reinterpret_cast<Element const*>(params.ptr_X);
    auto const* Wk_ptr = reinterpret_cast<Element const*>(params.ptr_Wk);
    auto const* Wv_ptr = reinterpret_cast<Element const*>(params.ptr_Wv);
    auto* Kc_ptr = reinterpret_cast<Element*>(params.ptr_Kc);
    auto* Vc_ptr = reinterpret_cast<Element*>(params.ptr_Vc);

    const int n_start_new = n_block_new * kBlockN;
    Element const* gX  = X_ptr + (size_t)batch * S * H + (size_t)n_start_new * H;
    Element const* gWk = Wk_ptr + (size_t)head * D * H;
    Element const* gWv = Wv_ptr + (size_t)head * D * H;

    // Smem: sK/sWk(aliased) + sV/sWv(aliased) + sX
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto align128 = [](uintptr_t p) -> uintptr_t { return (p + 127) & ~uintptr_t(127); };
    uintptr_t base = reinterpret_cast<uintptr_t>(smem_raw);
    base = align128(base);
    Element* sK_ptr = reinterpret_cast<Element*>(base);
    Element* sWk_ptr = sK_ptr;
    base = align128(base + cute::cosize(SmemLayoutK{}) * sizeof(Element));
    Element* sV_ptr = reinterpret_cast<Element*>(base);
    Element* sWv_ptr = sV_ptr;
    base = align128(base + cute::cosize(SmemLayoutV{}) * sizeof(Element));
    Element* sX_ptr = reinterpret_cast<Element*>(base);

    Tensor sK  = make_tensor(make_smem_ptr(sK_ptr),  SmemLayoutK{});
    Tensor sV  = make_tensor(make_smem_ptr(sV_ptr),  SmemLayoutV{});
    Tensor sX  = make_tensor(make_smem_ptr(sX_ptr),  SmemLayoutX{});
    Tensor sWk = make_tensor(make_smem_ptr(sWk_ptr), SmemLayoutW{});
    Tensor sWv = make_tensor(make_smem_ptr(sWv_ptr), SmemLayoutW{});

    auto bar_mma = [](){ asm volatile("bar.sync 1, 128;\n" ::); };

    TiledMmaProj tiled_mma_proj;
    auto thr_mma = tiled_mma_proj.get_thread_slice(tid);
    Tensor acc_k = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_v = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    clear(acc_k); clear(acc_v);

    Tensor tCsX  = thr_mma.partition_fragment_A(sX);
    Tensor tCsWk = thr_mma.partition_fragment_B(sWk);
    Tensor tCsWv = thr_mma.partition_fragment_B(sWv);

    constexpr int kElemsPerVec = 8;

    #pragma unroll
    for (int cs = 0; cs < kNumProjChunks; ++cs) {
        // Load X chunk
        {
            constexpr int kVecsPerRow = kHiddenChunk / kElemsPerVec;
            constexpr int kTotalVecs = kBlockN * kVecsPerRow;
            constexpr int kVecsPerThread = kTotalVecs / kNThreadsMMA;
            #pragma unroll
            for (int v = 0; v < kVecsPerThread; ++v) {
                int idx = v * kNThreadsMMA + tid;
                int row = idx / kVecsPerRow;
                int col = (idx % kVecsPerRow) * kElemsPerVec;
                Element const* g_src = gX + (size_t)row * H + cs * kHiddenChunk + col;
                uint32_t s_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sX(row, col)));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s_addr), "l"(g_src));
            }
        }
        // Load Wk chunk
        {
            constexpr int kVecsPerRow = kHiddenChunk / kElemsPerVec;
            constexpr int kTotalVecs = kHeadDim * kVecsPerRow;
            constexpr int kVecsPerThread = kTotalVecs / kNThreadsMMA;
            #pragma unroll
            for (int v = 0; v < kVecsPerThread; ++v) {
                int idx = v * kNThreadsMMA + tid;
                int row = idx / kVecsPerRow;
                int col = (idx % kVecsPerRow) * kElemsPerVec;
                Element const* g_src = gWk + (size_t)row * H + cs * kHiddenChunk + col;
                uint32_t s_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sWk(row, col)));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s_addr), "l"(g_src));
            }
        }
        // Load Wv chunk
        {
            constexpr int kVecsPerRow = kHiddenChunk / kElemsPerVec;
            constexpr int kTotalVecs = kHeadDim * kVecsPerRow;
            constexpr int kVecsPerThread = kTotalVecs / kNThreadsMMA;
            #pragma unroll
            for (int v = 0; v < kVecsPerThread; ++v) {
                int idx = v * kNThreadsMMA + tid;
                int row = idx / kVecsPerRow;
                int col = (idx % kVecsPerRow) * kElemsPerVec;
                Element const* g_src = gWv + (size_t)row * H + cs * kHiddenChunk + col;
                uint32_t s_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sWv(row, col)));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s_addr), "l"(g_src));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        bar_mma();
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

    // Write to sK/sV
    Tensor tCsK = thr_mma.partition_C(sK);
    Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tCcV = thr_mma.partition_C(cV);
    #pragma unroll
    for (int i = 0; i < size(acc_k); ++i) {
        tCsK(i) = Element(acc_k(i));
        sV(get<1>(tCcV(i)), get<0>(tCcV(i))) = Element(acc_v(i));
    }
    bar_mma();

    // Store to Kc/Vc
    constexpr int kVecsPerRowKV = kHeadDim / kElemsPerVec;
    constexpr int kTotalVecsKV = kBlockN * kVecsPerRowKV;
    constexpr int kVecsPerThreadKV = kTotalVecsKV / kNThreadsMMA;
    #pragma unroll
    for (int v = 0; v < kVecsPerThreadKV; ++v) {
        int idx = v * kNThreadsMMA + tid;
        int i = idx / kVecsPerRowKV;
        int d = (idx % kVecsPerRowKV) * kElemsPerVec;
        int row_new = n_start_new + i;
        if (row_new < S) {
            size_t off = ((size_t)batch * S + row_new) * NH * D + (size_t)head * D + d;
            uint4 kv_k, kv_v;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                reinterpret_cast<Element*>(&kv_k)[k] = sK(i, d + k);
                reinterpret_cast<Element*>(&kv_v)[k] = sV(d + k, i);
            }
            *reinterpret_cast<uint4*>(Kc_ptr + off) = kv_k;
            *reinterpret_cast<uint4*>(Vc_ptr + off) = kv_v;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Pass 2: Q-centric attention — reads K/V from Kc/Vc and Kpast/Vpast
// Grid: (num_m_blocks, num_heads, batch)
////////////////////////////////////////////////////////////////////////////////
template <typename Traits>
__global__ __launch_bounds__(Traits::kNThreadsMMA, 2)
void qcentric_attn_kernel(TwoPassParams params)
{
    using Element = typename Traits::Element;
    constexpr int kBlockM   = Traits::kBlockM;
    constexpr int kBlockN   = Traits::kBlockN;
    constexpr int kHeadDim  = Traits::kHeadDim;
    constexpr int kNThreadsMMA = Traits::kNThreadsMMA;
    using SmemLayoutQ = typename Traits::SmemLayoutQ;
    using SmemLayoutK = typename Traits::SmemLayoutK;
    using SmemLayoutV = typename Traits::SmemLayoutV;
    using TiledMmaQK = typename Traits::TiledMmaQK;
    using TiledMmaPV = typename Traits::TiledMmaPV;

    const int m_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int tid     = threadIdx.x;

    const int S = params.seqlen;
    const int D = params.head_dim;
    const int NH = params.num_heads;
    const int S_past = params.seqlen_past;
    const int m_start = m_block * kBlockM;
    const int num_n_blocks_past = params.num_n_blocks_past;

    auto const* Q_ptr  = reinterpret_cast<Element const*>(params.ptr_Q);
    auto const* Kc_ptr = reinterpret_cast<Element const*>(params.ptr_Kc);
    auto const* Vc_ptr = reinterpret_cast<Element const*>(params.ptr_Vc);
    auto const* Kpast_ptr = reinterpret_cast<Element const*>(params.ptr_Kpast);
    auto const* Vpast_ptr = reinterpret_cast<Element const*>(params.ptr_Vpast);
    auto* Op_ptr = reinterpret_cast<Element*>(params.ptr_Opart);
    float* LSE_ptr = params.ptr_LSE;

    // Smem: sQ + sK + sV
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto align128 = [](uintptr_t p) -> uintptr_t { return (p + 127) & ~uintptr_t(127); };
    uintptr_t base = reinterpret_cast<uintptr_t>(smem_raw);
    base = align128(base);
    Element* sQ_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));
    Element* sK_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutK{}) * sizeof(Element));
    Element* sV_ptr = reinterpret_cast<Element*>(base);

    Tensor sQ = make_tensor(make_smem_ptr(sQ_ptr), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(sK_ptr), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutV{});

    auto bar_mma = [](){ asm volatile("bar.sync 1, 128;\n" ::); };

    // Load Q once via cp.async
    {
        constexpr int kElemsPerVec = 8;
        constexpr int kVecsPerRow = kHeadDim / kElemsPerVec;
        constexpr int kTotalVecs = kBlockM * kVecsPerRow;
        constexpr int kVecsPerThread = kTotalVecs / kNThreadsMMA;
        Element const* gQ = Q_ptr + ((size_t)batch * S * NH + m_start * NH + head) * D;
        #pragma unroll
        for (int v = 0; v < kVecsPerThread; ++v) {
            int idx = v * kNThreadsMMA + tid;
            int row = idx / kVecsPerRow;
            int col = (idx % kVecsPerRow) * kElemsPerVec;
            Element const* g_src = gQ + (size_t)row * NH * D + col;
            uint32_t s_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sQ(row, col)));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s_addr), "l"(g_src));
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        bar_mma();
    }

    // Attention setup
    float const softmax_scale_log2 = params.softmax_scale * float(M_LOG2E);
    constexpr int kNRows = 2 * (kBlockM / 64);
    using SoftmaxT = flash::Softmax<kNRows>;
    SoftmaxT softmax(softmax_scale_log2);

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

    Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
    Tensor tOrV = thr_mma_pv.partition_fragment_B(sV);

    Tensor acc_o = partition_fragment_C(tiled_mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);

    // Causal limit
    const int max_n_block = params.is_causal
        ? min(params.num_n_blocks, num_n_blocks_past + m_block + 1)
        : params.num_n_blocks;

    // KV load helper
    constexpr int kElemsPerVec = 8;
    constexpr int kVecsPerRow = kHeadDim / kElemsPerVec;
    constexpr int kTotalVecs = kBlockN * kVecsPerRow;
    constexpr int kVecsPerThread = kTotalVecs / kNThreadsMMA;

    auto load_kv = [&](Element const* gK, Element const* gV) {
        #pragma unroll
        for (int v = 0; v < kVecsPerThread; ++v) {
            int idx = v * kNThreadsMMA + tid;
            int i = idx / kVecsPerRow;
            int d = (idx % kVecsPerRow) * kElemsPerVec;
            uint4 vk = *reinterpret_cast<uint4 const*>(gK + (size_t)i * NH * D + d);
            uint4 vv = *reinterpret_cast<uint4 const*>(gV + (size_t)i * NH * D + d);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                sK(i, d + k) = reinterpret_cast<Element const*>(&vk)[k];
                sV(d + k, i) = reinterpret_cast<Element const*>(&vv)[k];
            }
        }
    };

    // KV load for past (different layout: [B, S_past, NH, D])
    auto load_kv_past = [&](int n_start) {
        #pragma unroll
        for (int v = 0; v < kVecsPerThread; ++v) {
            int idx = v * kNThreadsMMA + tid;
            int i = idx / kVecsPerRow;
            int d = (idx % kVecsPerRow) * kElemsPerVec;
            size_t off = ((size_t)batch * S_past + n_start + i) * NH * D + (size_t)head * D + d;
            uint4 vk = *reinterpret_cast<uint4 const*>(Kpast_ptr + off);
            uint4 vv = *reinterpret_cast<uint4 const*>(Vpast_ptr + off);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                sK(i, d + k) = reinterpret_cast<Element const*>(&vk)[k];
                sV(d + k, i) = reinterpret_cast<Element const*>(&vv)[k];
            }
        }
    };

    // Main KV loop
    for (int n_block = 0; n_block < max_n_block; ++n_block) {
        const int n_start = n_block * kBlockN;
        const bool is_past = (n_block < num_n_blocks_past);

        if (is_past) {
            load_kv_past(n_start);
        } else {
            int n_start_new = n_start - S_past;
            Element const* gK = Kc_ptr + ((size_t)batch * S + n_start_new) * NH * D + (size_t)head * D;
            Element const* gV = Vc_ptr + ((size_t)batch * S + n_start_new) * NH * D + (size_t)head * D;
            load_kv(gK, gV);
        }
        bar_mma();

        // QK GEMM
        Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
        flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK, acc_s);
        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(acc_s);

        // Causal mask
        if (params.is_causal) {
            auto thread0_mma = TiledMmaQK{}.get_thread_slice(_0{});
            Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
            Tensor tScS = thr_mma_qk.partition_C(cS);
            Tensor t0ScS = thread0_mma.partition_C(cS);
            Tensor acc_s_rc = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Tensor tScS_rc = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol(tScS.layout()));
            Tensor t0ScS_rc = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol(t0ScS.layout()));
            int thread_col_offset = get<1>(tScS_rc(_0{}, _0{}));
            int causal_row_offset = S_past + 1 - n_start + m_start - thread_col_offset;
            #pragma unroll
            for (int m = 0; m < size<0>(acc_s_rc); ++m) {
                int row_rel = get<0>(tScS_rc(m, _0{}));
                int col_limit = row_rel + causal_row_offset;
                #pragma unroll
                for (int n = 0; n < size<1>(acc_s_rc); ++n) {
                    int col_rel_t0 = get<1>(t0ScS_rc(_0{}, n));
                    if (col_rel_t0 >= col_limit) { acc_s_rc(m, n) = -INFINITY; }
                }
            }
        }

        // Online softmax
        bool is_first = (n_block == 0);
        if (is_first) {
            (void)softmax.template max_get_scale<true, true>(acc_s);
            softmax.template online_softmax<true, true>(acc_s);
        } else {
            auto scores_scale = softmax.template max_get_scale<false, true>(acc_s);
            softmax.rescale_o(acc_o, scores_scale);
            softmax.template online_softmax<false, true>(acc_s);
        }

        // Convert to fp16 P
        Tensor tOrP_acc = make_tensor(acc_s.data(),
            flash::convert_layout_acc_Aregs<TiledMmaPV>(acc_s.layout()));
        Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
        flash::convert_type_out(tOrP_acc, tOrP);

        // PV GEMM
        if (is_first) {
            flash::gemm<true, -1>(tiled_mma_pv, tOrP, tOrV, acc_o);
        } else {
            flash::gemm<false, -1>(tiled_mma_pv, tOrP, tOrV, acc_o);
        }
        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(acc_o);
    }

    // Finalize softmax and write output to Opart[n=0]
    auto final_scale = softmax.finalize();
    softmax.rescale_o(acc_o, final_scale);

    const int num_n_alloc = params.num_n_blocks;

    // Write to Opart[:,:,:,0,:]
    {
        Tensor cO = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tOcO = thr_mma_pv.partition_C(cO);
        #pragma unroll
        for (int i = 0; i < size(acc_o); ++i) {
            int row_rel = get<0>(tOcO(i));
            int col = get<1>(tOcO(i));
            int row = m_start + row_rel;
            if (row < S) {
                size_t off = ((((size_t)batch * S + row) * NH + head) * num_n_alloc + 0) * D + col;
                Op_ptr[off] = Element(acc_o(i));
            }
        }
    }

    // Set LSE: n=0 -> 0.0, n>0 -> -INF
    for (int row_off = tid; row_off < kBlockM; row_off += kNThreadsMMA) {
        int row = m_start + row_off;
        if (row < S) {
            size_t base_lse = (((size_t)batch * S + row) * NH + head) * num_n_alloc;
            LSE_ptr[base_lse] = 0.0f;
            for (int n = 1; n < num_n_alloc; ++n) {
                LSE_ptr[base_lse + n] = -INFINITY;
            }
        }
    }
}

} // namespace proj_fused
