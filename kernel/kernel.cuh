/******************************************************************************
 * Projection-Fused Flash Attention — Kernel (SM90)
 *
 * Phase 2a (full): WGMMA for K/V projection AND attention (Q·K^T, P·V).
 *
 * 128 compute threads (1 warpgroup) + 32 storer threads = 160 threads/block.
 * Grid: (num_n_blocks, num_heads, batch).
 ******************************************************************************/
#pragma once

#include <cuda.h>       // CUtensorMap
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
#include "softmax.h"  // flash::Softmax
#include "utils.h"    // flash::gemm, flash::convert_layout_acc_Aregs, flash::convert_type_out

namespace proj_fused {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////
// Params
////////////////////////////////////////////////////////////////////////////////

struct ProjFusedParams {
    void const* __restrict__ ptr_X;   // [B, S_new, H]
    void const* __restrict__ ptr_Wk;  // [num_heads, D, H]
    void const* __restrict__ ptr_Wv;  // [num_heads, D, H]
    void const* __restrict__ ptr_Q;   // [B, S_new, num_heads, D]
    void*       __restrict__ ptr_Opart;  // [B, num_n_blocks_total, S_new, num_heads, D]
    void*       __restrict__ ptr_Kc;  // [B, S_new, num_heads, D] — newly-projected K (written)
    void*       __restrict__ ptr_Vc;  // [B, S_new, num_heads, D] — newly-projected V (written)
    float*      __restrict__ ptr_LSE; // [B, num_n_blocks_total, num_heads, S_new]
    // KV cache extension: past K,V stored in HBM.
    // Layout: [B, S_past, num_heads, D] — same stride scheme as Kc/Vc.
    void const* __restrict__ ptr_Kpast;  // nullptr if no past KV
    void const* __restrict__ ptr_Vpast;  // nullptr if no past KV
    CUtensorMap const* tma_desc_Q;      // 4D TMA descriptor for Q in global memory

    int batch, seqlen, num_heads, hidden_dim, head_dim;
    int seqlen_past;       // 0 if no past KV; MUST be multiple of kBlockN when nonzero
    int num_n_blocks_past; // seqlen_past / kBlockN
    int num_n_blocks;      // total (past + new)
    int num_m_blocks;      // over seqlen (S_new)
    float softmax_scale;
    int is_causal;
};

////////////////////////////////////////////////////////////////////////////////
// Main kernel
////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
__global__ __launch_bounds__(Traits::kNThreadsMMA, 2)
void proj_fused_fwd_kernel(ProjFusedParams params)
{
    using Element = typename Traits::Element;
    constexpr int kBlockM   = Traits::kBlockM;
    constexpr int kBlockN   = Traits::kBlockN;
    constexpr int kHeadDim  = Traits::kHeadDim;
    constexpr int kHiddenChunk = Traits::kHiddenChunk;
    constexpr int kNumProjChunks = Traits::kNumProjChunks;
    constexpr int kNThreadsMMA = Traits::kNThreadsMMA;
    using SmemLayoutQ = typename Traits::SmemLayoutQ;
    using SmemLayoutK = typename Traits::SmemLayoutK;
    using SmemLayoutV = typename Traits::SmemLayoutV;
    using SmemLayoutX = typename Traits::SmemLayoutX;
    using SmemLayoutW = typename Traits::SmemLayoutW;
    using TiledMmaProj = typename Traits::TiledMmaProj;
    using TiledMmaQK = typename Traits::TiledMmaQK;
    using TiledMmaPV = typename Traits::TiledMmaPV;

    const int n_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int tid     = threadIdx.x;

    const int S = params.seqlen;
    const int H = params.hidden_dim;
    const int D = params.head_dim;
    const int NH = params.num_heads;

    const int n_start = n_block * kBlockN;
    // KV cache extension: past n_blocks read directly from Kpast/Vpast; new n_blocks
    // project X→K,V on the fly. S_past (params.seqlen_past) is a multiple of kBlockN.
    const int S_past = params.seqlen_past;
    const int num_n_blocks_past = params.num_n_blocks_past;
    const bool is_past_kv = (n_block < num_n_blocks_past);
    const int n_start_new = n_start - S_past;  // local row in X_new for new n_blocks

    // Pointers
    auto const* X_ptr  = reinterpret_cast<Element const*>(params.ptr_X);
    auto const* Wk_ptr = reinterpret_cast<Element const*>(params.ptr_Wk);
    auto const* Wv_ptr = reinterpret_cast<Element const*>(params.ptr_Wv);
    auto const* Q_ptr  = reinterpret_cast<Element const*>(params.ptr_Q);
    auto* Op_ptr = reinterpret_cast<Element*>(params.ptr_Opart);
    auto* Kc_ptr = reinterpret_cast<Element*>(params.ptr_Kc);
    auto* Vc_ptr = reinterpret_cast<Element*>(params.ptr_Vc);
    float* LSE_ptr = params.ptr_LSE;
    auto const* Kpast_ptr = reinterpret_cast<Element const*>(params.ptr_Kpast);
    auto const* Vpast_ptr = reinterpret_cast<Element const*>(params.ptr_Vpast);

    // Base offsets for this (batch, head, n_block) — only valid for non-past blocks
    Element const* gX  = is_past_kv ? nullptr
                                    : X_ptr + (size_t)batch * S * H + (size_t)n_start_new * H;
    Element const* gWk = Wk_ptr + (size_t)head * D * H;
    Element const* gWv = Wv_ptr + (size_t)head * D * H;

    // ------------------------------------------------------------------
    // Shared memory layout (all dynamically allocated, aligned to 128B).
    // Aliased buffers — projection and attention phases don't overlap:
    //   sX  aliases first 8KB of sQ0  (projection only)
    //   sWk aliases sO0               (projection only, attention uses sO0)
    //   sWv aliases sO1               (projection only, attention uses sO1)
    // This saves ~40KB (136KB → 96KB).
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto align128 = [](uintptr_t p) -> uintptr_t { return (p + 127) & ~uintptr_t(127); };
    uintptr_t base = reinterpret_cast<uintptr_t>(smem_raw);
    base = align128(base);
    Element* sQ0_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));
    Element* sQ1_ptr = reinterpret_cast<Element*>(base);   // double-buffer
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));
    Element* sK_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutK{}) * sizeof(Element));
    Element* sV_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutV{}) * sizeof(Element));
    // sO0 and sO1 are the last two allocations — they alias sWk and sWv.
    Element* sO0_ptr = reinterpret_cast<Element*>(base);  // aliases sWk during projection
    Element* sWk_ptr = sO0_ptr;                           // same memory as sO0
    base = align128(base + cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));
    Element* sO1_ptr = reinterpret_cast<Element*>(base);  // aliases sWv during projection
    Element* sWv_ptr = sO1_ptr;                           // same memory as sO1
    // sX aliases the first 8KB of sQ0 (sX cosize ≤ sQ0 cosize, used only during projection).
    Element* sX_ptr = sQ0_ptr;

    using SmemLayoutO = typename Traits::SmemLayoutO;
    Tensor sQ0 = make_tensor(make_smem_ptr(sQ0_ptr), SmemLayoutQ{});
    Tensor sQ1 = make_tensor(make_smem_ptr(sQ1_ptr), SmemLayoutQ{});
    Tensor sK  = make_tensor(make_smem_ptr(sK_ptr),  SmemLayoutK{});
    Tensor sV  = make_tensor(make_smem_ptr(sV_ptr),  SmemLayoutV{});
    Tensor sX  = make_tensor(make_smem_ptr(sX_ptr),  SmemLayoutX{});
    Tensor sWk = make_tensor(make_smem_ptr(sWk_ptr), SmemLayoutW{});
    Tensor sWv = make_tensor(make_smem_ptr(sWv_ptr), SmemLayoutW{});
    Tensor sO0 = make_tensor(make_smem_ptr(sO0_ptr), SmemLayoutO{});
    Tensor sO1 = make_tensor(make_smem_ptr(sO1_ptr), SmemLayoutO{});

    // ------------------------------------------------------------------
    // Barrier: 128 MMA threads only (storer threads don't participate).
    // ------------------------------------------------------------------
    auto bar_mma = [](){ asm volatile("bar.sync 1, 128;\n" ::); };

    // ------------------------------------------------------------------
    // K/V acquisition: either load from past KV cache (is_past_kv) OR
    // project from X·W^T via WGMMA (new tokens).
    // ------------------------------------------------------------------
    if (is_past_kv) {
        // Load K[n_block] from Kpast, V[n_block] from Vpast into sK, sV.
        // Kpast layout: [B, S_past, NH, D], row-major in (d) dim.
        // We need sK (K-major, [kBlockN, kHeadDim]) and sV (MN-major, [kHeadDim, kBlockN] transposed).
        constexpr int kElemsPerVec = 8;
        constexpr int kVecsPerRow = kHeadDim / kElemsPerVec;   // 16
        constexpr int kTotalVecs = kBlockN * kVecsPerRow;       // 1024
        constexpr int kVecsPerThread = kTotalVecs / kNThreadsMMA;
        #pragma unroll
        for (int v = 0; v < kVecsPerThread; ++v) {
            int idx = v * kNThreadsMMA + tid;
            int i = idx / kVecsPerRow;
            int d = (idx % kVecsPerRow) * kElemsPerVec;
            int row = n_start + i;
            size_t off = ((size_t)batch * S_past + row) * NH * D + (size_t)head * D + d;
            Element const* gK = Kpast_ptr + off;
            Element const* gV = Vpast_ptr + off;
            // Load 8 fp16 from HBM, store to smem via swizzled accessor.
            uint4 vk = *reinterpret_cast<uint4 const*>(gK);
            uint4 vv = *reinterpret_cast<uint4 const*>(gV);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                // Individual fp16 stores — sK K-major, sV MN-major (V^T in smem)
                Element kv = reinterpret_cast<Element const*>(&vk)[k];
                Element vv2 = reinterpret_cast<Element const*>(&vv)[k];
                sK(i, d + k) = kv;
                sV(d + k, i) = vv2;  // transposed layout
            }
        }
    } else {
        TiledMmaProj tiled_mma_proj;
        auto thr_mma = tiled_mma_proj.get_thread_slice(tid);

        Tensor acc_k = partition_fragment_C(tiled_mma_proj,
            Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor acc_v = partition_fragment_C(tiled_mma_proj,
            Shape<Int<kBlockN>, Int<kHeadDim>>{});
        clear(acc_k);
        clear(acc_v);

        Tensor tCsX  = thr_mma.partition_fragment_A(sX);
        Tensor tCsWk = thr_mma.partition_fragment_B(sWk);
        Tensor tCsWv = thr_mma.partition_fragment_B(sWv);

        constexpr int kElemsPerVec = 8;
        static_assert(kHiddenChunk % kElemsPerVec == 0, "");

        auto load_X_chunk = [&](Element const* gSrc, int cs) {
            constexpr int kVecsPerRow = kHiddenChunk / kElemsPerVec;
            constexpr int kTotalVecs = kBlockN * kVecsPerRow;
            constexpr int kVecsPerThread = kTotalVecs / kNThreadsMMA;
            #pragma unroll
            for (int v = 0; v < kVecsPerThread; ++v) {
                int idx = v * kNThreadsMMA + tid;
                int row = idx / kVecsPerRow;
                int col = (idx % kVecsPerRow) * kElemsPerVec;
                Element const* g_src = gSrc + (size_t)row * H + cs * kHiddenChunk + col;
                Element* s_dst = &sX(row, col);
                uint32_t s_dst_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(s_dst));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(s_dst_u32), "l"(g_src));
            }
        };
        auto load_W_chunk = [&](Element const* gSrc, int cs, auto& sDst) {
            constexpr int kVecsPerRow = kHiddenChunk / kElemsPerVec;
            constexpr int kTotalVecs = kHeadDim * kVecsPerRow;
            constexpr int kVecsPerThread = kTotalVecs / kNThreadsMMA;
            #pragma unroll
            for (int v = 0; v < kVecsPerThread; ++v) {
                int idx = v * kNThreadsMMA + tid;
                int row = idx / kVecsPerRow;
                int col = (idx % kVecsPerRow) * kElemsPerVec;
                Element const* g_src = gSrc + (size_t)row * H + cs * kHiddenChunk + col;
                Element* s_dst = &sDst(row, col);
                uint32_t s_dst_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(s_dst));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(s_dst_u32), "l"(g_src));
            }
        };

        #pragma unroll
        for (int cs = 0; cs < kNumProjChunks; ++cs) {
            load_X_chunk(gX, cs);
            load_W_chunk(gWk, cs, sWk);
            load_W_chunk(gWv, cs, sWv);
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 0;\n" ::);
            bar_mma();
            bool first = (cs == 0);
            if (first) {
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

        Tensor tCsK = thr_mma.partition_C(sK);
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tCcV = thr_mma.partition_C(cV);
        #pragma unroll
        for (int i = 0; i < size(acc_k); ++i) {
            tCsK(i) = Element(acc_k(i));
            int r = get<0>(tCcV(i));
            int c = get<1>(tCcV(i));
            sV(c, r) = Element(acc_v(i));
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // KV cache store: MMA threads write sK, sV -> HBM (replaces storer warp).
    // With 128 threads (vs 32 storer), uses uint4 stores for better throughput.
    // ------------------------------------------------------------------
    if (!is_past_kv) {
        constexpr int kElemsPerVecKV = 8;
        constexpr int kVecsPerRowKV = kHeadDim / kElemsPerVecKV;   // 16
        constexpr int kTotalVecsKV = kBlockN * kVecsPerRowKV;      // 1024
        constexpr int kVecsPerThreadKV = kTotalVecsKV / kNThreadsMMA;  // 8
        #pragma unroll
        for (int v = 0; v < kVecsPerThreadKV; ++v) {
            int idx = v * kNThreadsMMA + tid;
            int i = idx / kVecsPerRowKV;
            int d = (idx % kVecsPerRowKV) * kElemsPerVecKV;
            int row_new = n_start_new + i;
            if (row_new < S) {
                size_t off = ((size_t)batch * S + row_new) * NH * D + (size_t)head * D + d;
                // Read 8 fp16 from swizzled sK, pack into uint4, write to HBM.
                uint4 kv_k, kv_v;
                Element* kk = reinterpret_cast<Element*>(&kv_k);
                Element* vv = reinterpret_cast<Element*>(&kv_v);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    kk[k] = sK(i, d + k);
                    vv[k] = sV(d + k, i);
                }
                asm volatile("st.cs.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
                    :: "l"(Kc_ptr + off), "r"(kv_k.x), "r"(kv_k.y), "r"(kv_k.z), "r"(kv_k.w)
                    : "memory");
                asm volatile("st.cs.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
                    :: "l"(Vc_ptr + off), "r"(kv_v.x), "r"(kv_v.y), "r"(kv_v.z), "r"(kv_v.w)
                    : "memory");
            }
        }
    }

    // ------------------------------------------------------------------
    // Attention loop: WGMMA Q·K^T, softmax, WGMMA P·V.
    // ------------------------------------------------------------------

    // Softmax scale: log2(e) * scale
    float const softmax_scale_log2 = params.softmax_scale * float(M_LOG2E);

    // kNRows = 2 * MMA_M for SM90 WGMMA.  For kBlockM=64, MMA_M=1 -> kNRows=2.
    constexpr int kNRows = 2 * (kBlockM / 64);
    using SoftmaxT = flash::Softmax<kNRows, /*Max_offset=*/0>;

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

    // Partitioned SS operands for QK GEMM (per-buffer).
    Tensor tSrQ0 = thr_mma_qk.partition_fragment_A(sQ0);
    Tensor tSrQ1 = thr_mma_qk.partition_fragment_A(sQ1);
    Tensor tSrK  = thr_mma_qk.partition_fragment_B(sK);
    // Partitioned operands for PV: A in registers (RS-mode), B in smem (MN-major V)
    Tensor tOrV = thr_mma_pv.partition_fragment_B(sV);

    // Pre-partition STSM copy infrastructure for both sO buffers.
    // Building this ONCE outside the attention loop avoids per-iteration tensor
    // construction overhead (exp6 lesson: Tensor::make_tensor inside loop is costly).
    auto smem_tiled_copy_O = make_tiled_copy_C(
        cute::Copy_Atom<cute::SM90_U32x4_STSM_N, Element>{}, tiled_mma_pv);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tid);
    Tensor taccOsO0 = smem_thr_copy_O.partition_D(sO0);
    Tensor taccOsO1 = smem_thr_copy_O.partition_D(sO1);
    const int num_n_blocks_o = params.num_n_blocks;
    constexpr int kElemsPerVecO = 8;
    constexpr int kVecsPerRowO = kHeadDim / kElemsPerVecO;       // 16
    constexpr int kTotalVecsO  = kBlockM * kVecsPerRowO;         // 64*16=1024
    constexpr int kVecsPerThreadO = kTotalVecsO / kNThreadsMMA;  // 8

    // ---- TMA Q-loader: replaces cp.async with hardware TMA ----
    // mbarriers in dynamic smem (after sO1), for double-buffered TMA completion.
    // Each sQ half is kBlockM*64*sizeof(Element)=8192 bytes. Total per tile: 16384 bytes.
    constexpr int kTmaHalfBytes = kBlockM * 64 * (int)sizeof(Element);  // 8192
    // Mbarrier storage at end of dynamic smem.
    uint64_t* mbar_Q = reinterpret_cast<uint64_t*>(
        reinterpret_cast<uintptr_t>(sO1_ptr) +
        cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));
    mbar_Q = reinterpret_cast<uint64_t*>((reinterpret_cast<uintptr_t>(mbar_Q) + 7) & ~uintptr_t(7));

    if (tid == 0) {
        uint32_t mb0 = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_Q[0]));
        uint32_t mb1 = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_Q[1]));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mb0), "r"(1));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mb1), "r"(1));
    }
    bar_mma();

    CUtensorMap const* tma_desc = params.tma_desc_Q;
    int tma_phase[2] = {0, 0};

    // TMA load Q: thread 0 issues 2 × cp.async.bulk.tensor.4d (each 8KB).
    // 4D coords: {d_offset, head, m_start, batch}
    auto tma_load_Q = [&](int m_start, int buf) {
        if (tid == 0) {
            Element* sq = (buf == 0) ? sQ0_ptr : sQ1_ptr;
            uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(sq));
            uint32_t s1 = s0 + kTmaHalfBytes;  // second d-half at +8192 bytes
            uint32_t mb = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_Q[buf]));
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                         :: "r"(mb), "r"(2 * kTmaHalfBytes));
            // Load d=[0,63]
            asm volatile(
                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
                :: "r"(s0), "l"(tma_desc),
                   "r"(0), "r"(head), "r"(m_start), "r"(batch),
                   "r"(mb));
            // Load d=[64,127]
            asm volatile(
                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
                :: "r"(s1), "l"(tma_desc),
                   "r"(64), "r"(head), "r"(m_start), "r"(batch),
                   "r"(mb));
        }
    };

    auto tma_wait_Q = [&](int buf) {
        uint32_t mb = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_Q[buf]));
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "WAIT_%=:\n"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
            "@!P bra WAIT_%=;\n"
            "}\n"
            :: "r"(mb), "r"(tma_phase[buf]));
        tma_phase[buf] ^= 1;
    };

    // Causal skip: for causal attention, new KV block at local index new_n_idx only
    // contributes to Q blocks m >= new_n_idx. Past KV blocks (n_block < num_n_blocks_past)
    // always precede all Q positions, so no skipping needed there.
    // new_n_idx = n_block - num_n_blocks_past (0-based among new KV blocks).
    const int new_n_idx = n_block - num_n_blocks_past;
    const int m_start_block = (params.is_causal && new_n_idx > 0) ? new_n_idx : 0;

    // Write LSE=-INF for entirely-masked Q blocks (skipped below).
    // LSE layout: [B, S, NH, N] — contiguous in N dimension.
    {
        const int num_n_blocks_lse = params.num_n_blocks;
        for (int m_block = 0; m_block < m_start_block; ++m_block) {
            int m_start = m_block * kBlockM;
            for (int i = tid; i < kBlockM; i += kNThreadsMMA) {
                int row = m_start + i;
                size_t off_lse = (((size_t)batch * S + row) * NH + head) * num_n_blocks_lse
                               + n_block;
                LSE_ptr[off_lse] = -INFINITY;
            }
        }
    }

    // Prefetch first Q block via TMA.
    if (m_start_block < params.num_m_blocks) {
        tma_load_Q(m_start_block * kBlockM, m_start_block & 1);
    }

    for (int m_block = m_start_block; m_block < params.num_m_blocks; ++m_block) {
        int m_start = m_block * kBlockM;
        int buf = m_block & 1;

        // Issue TMA prefetch for m_block+1 into the other buffer.
        if (m_block + 1 < params.num_m_blocks) {
            tma_load_Q((m_block + 1) * kBlockM, buf ^ 1);
        }
        // Wait for current Q tile's TMA completion.
        tma_wait_Q(buf);
        bar_mma();

        // GEMM-I: S = Q · K^T  (wg_wait=-1: async, lets stores overlap with tensor cores)
        Tensor acc_s = partition_fragment_C(tiled_mma_qk,
            Shape<Int<kBlockM>, Int<kBlockN>>{});
        if (buf == 0) {
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ0, tSrK, acc_s);
        } else {
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ1, tSrK, acc_s);
        }

        // OVERLAP: gmem stores from previous m_block while QK GEMM runs on tensor cores.
        // sO_prev = sO[(m_block-1)&1] = the OTHER double-buffer slot from last iteration.
        // Safety: bar_mma#2 of m-1 (completed before bar_mma#1 above) guarantees all
        // STSM writes to sO_prev are visible; QK only reads sQ[buf] and sK (not sO).
        // Opart layout: [B, S, NH, N, D] — contiguous in D, then N.
        if (m_block > m_start_block) {
            auto& sO_prev = (buf == 0) ? sO1 : sO0;
            const int m_prev_start = m_start - kBlockM;
            #pragma unroll
            for (int v = 0; v < kVecsPerThreadO; ++v) {
                int idx = v * kNThreadsMMA + tid;
                int i = idx / kVecsPerRowO;
                int c = (idx % kVecsPerRowO) * kElemsPerVecO;
                int row = m_prev_start + i;
                size_t off = ((((size_t)batch * S + row) * NH + head) * num_n_blocks_o
                           + n_block) * D + c;
                uint4 val = *reinterpret_cast<uint4 const*>(&sO_prev(i, c));
                // Streaming store: bypass L2 to avoid cache pollution from 1GB Opart writes
                asm volatile("st.cs.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
                    :: "l"(Op_ptr + off), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                    : "memory");
            }
        }
        // Wait for QK GEMM to complete and fence the accumulator.
        // Typically returns immediately (QK ~0.1 µs < stores ~0.6 µs).
        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(acc_s);

        // Causal mask
        if (params.is_causal) {
            auto thread0_mma = TiledMmaQK{}.get_thread_slice(_0{});
            Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
            Tensor tScS = thr_mma_qk.partition_C(cS);
            Tensor t0ScS = thread0_mma.partition_C(cS);
            Tensor acc_s_rc = make_tensor(acc_s.data(),
                flash::convert_layout_acc_rowcol(acc_s.layout()));
            Tensor tScS_rc = make_tensor(tScS.data(),
                flash::convert_layout_acc_rowcol(tScS.layout()));
            Tensor t0ScS_rc = make_tensor(t0ScS.data(),
                flash::convert_layout_acc_rowcol(t0ScS.layout()));
            int thread_col_offset = get<1>(tScS_rc(_0{}, _0{}));
            // With KV cache: Q row absolute = S_past + m_start + row_rel
            //                K col absolute = n_start + col_rel (n_start already covers past+new)
            // Mask when col_abs > row_abs  =>  col_rel_t0 >= row_rel + causal_row_offset
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

        // Softmax (online, first and only iteration per m-block in KV-centric layout)
        SoftmaxT softmax(softmax_scale_log2);
        (void)softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(acc_s);
        softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(acc_s);

        // Convert acc_s -> fp16 P in registers with A-operand layout for RS-mode PV.
        Tensor tOrP_acc = make_tensor(acc_s.data(),
            flash::convert_layout_acc_Aregs<TiledMmaPV>(acc_s.layout()));
        Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
        flash::convert_type_out(tOrP_acc, tOrP);

        // GEMM-II: O = P · V (RS mode) — issue with wg_wait=-1 so softmax finalize
        // and LSE stores can overlap with PV WGMMA (FA3 IntraWGOverlap pattern).
        Tensor acc_o = partition_fragment_C(tiled_mma_pv,
            Shape<Int<kBlockM>, Int<kHeadDim>>{});
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_pv, tOrP, tOrV, acc_o);

        // Finalize softmax while PV WGMMA runs in background.
        // finalize() does quad_allreduce_ on row_sum + stores LSE into row_sum,
        // and returns scores_scale = 1/row_sum. Touches softmax state only, NOT acc_o.
        auto scores_scale = softmax.finalize();

        // LSE write — also overlaps with PV (doesn't touch acc_o).
        // LSE layout: [B, S, NH, N] — contiguous in N dimension.
        {
            const int num_n_blocks_lse = params.num_n_blocks;
            Tensor cO_lse = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
            Tensor tOcO_lse = thr_mma_pv.partition_C(cO_lse);
            int lane = tid % 32;
            if (lane % 4 == 0) {
                Tensor tOcO_rc = make_tensor(tOcO_lse.data(),
                    flash::convert_layout_acc_rowcol(tOcO_lse.layout()));
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    int row_rel = get<0>(tOcO_rc(mi, _0{}));
                    int row = m_start + row_rel;
                    float lse = softmax.row_sum(mi);
                    size_t off_lse = (((size_t)batch * S + row) * NH + head)
                                   * num_n_blocks_lse + n_block;
                    LSE_ptr[off_lse] = lse;
                }
            }
        }

        // Wait for PV WGMMA to complete, then rescale acc_o.
        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(acc_o);
        softmax.rescale_o(acc_o, scores_scale);

        // Convert acc_o (fp32) → rO (fp16) register fragment, then STSM into sO.
        // Stores are moved to the START of the NEXT iteration (overlap with QK) so
        // only STSM + bar_mma #2 remain here.  bar_mma #2 is still required: it ensures
        // all threads' stmatrix writes are globally visible before the NEXT iteration
        // reads from this buffer when doing the overlapped stores.
        {
            Tensor rO = make_tensor_like<Element>(acc_o);
            flash::convert_type_out(acc_o, rO);
            Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
            auto& taccOsO_cur = (buf == 0) ? taccOsO0 : taccOsO1;
            cute::copy(smem_tiled_copy_O, taccOrO, taccOsO_cur);
            bar_mma();  // #2: sync stmatrix → enables next-iteration overlap store
        }
    }

    // Post-loop: gmem stores for the last m_block (no next-iteration QK to overlap with).
    // bar_mma #2 of the last iteration completed just before the loop exited, so
    // sO_last is fully written and safe to read.
    if (m_start_block < params.num_m_blocks) {
        const int last_m_start = (params.num_m_blocks - 1) * kBlockM;
        const int last_buf = (params.num_m_blocks - 1) & 1;
        auto& sO_last = (last_buf == 0) ? sO0 : sO1;
        #pragma unroll
        for (int v = 0; v < kVecsPerThreadO; ++v) {
            int idx = v * kNThreadsMMA + tid;
            int i = idx / kVecsPerRowO;
            int c = (idx % kVecsPerRowO) * kElemsPerVecO;
            int row = last_m_start + i;
            size_t off = ((((size_t)batch * S + row) * NH + head) * num_n_blocks_o
                       + n_block) * D + c;
            uint4 val = *reinterpret_cast<uint4 const*>(&sO_last(i, c));
            asm volatile("st.cs.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
                :: "l"(Op_ptr + off), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                : "memory");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Combine kernel
////////////////////////////////////////////////////////////////////////////////

struct CombineParams {
    void const* __restrict__ ptr_Opart;
    float const* __restrict__ ptr_LSE;
    void* __restrict__ ptr_O;
    int batch, seqlen, num_heads, head_dim, num_n_blocks;
};

template <typename Element, int kHeadDim>
__global__ void proj_fused_combine_kernel(CombineParams params)
{
    const int m_row = blockIdx.x;
    const int head  = blockIdx.y;
    const int batch = blockIdx.z;
    const int tid = threadIdx.x;

    const int S = params.seqlen;
    const int NH = params.num_heads;
    const int D = params.head_dim;
    const int N = params.num_n_blocks;

    auto const* Op = reinterpret_cast<Element const*>(params.ptr_Opart);
    auto const* LSE = params.ptr_LSE;
    auto* O = reinterpret_cast<Element*>(params.ptr_O);

    // Phase 1: Load all N LSE values into smem (N threads, 1 read each).
    // LSE layout: [B, S, NH, N] — N values for this (batch, m_row, head) are contiguous.
    extern __shared__ float smem_w[];
    if (tid < N) {
        size_t off = (((size_t)batch * S + m_row) * NH + head) * N + tid;
        smem_w[tid] = LSE[off];
    }
    __syncthreads();

    // Phase 2: Find max LSE from smem (all threads, cheap serial scan).
    float lse_max = -INFINITY;
    for (int n = 0; n < N; ++n) {
        float v = smem_w[n];
        if (v > lse_max) lse_max = v;
    }

    // Early exit: if lse_max is -INF, all blocks are masked → output zero.
    if (lse_max == -INFINITY) {
        for (int d = tid; d < kHeadDim; d += blockDim.x) {
            size_t off_final = ((size_t)batch * S + m_row) * NH * D + (size_t)head * D + d;
            O[off_final] = Element(0.f);
        }
        return;
    }

    // Phase 3: Precompute exp weights in smem.  Only N threads each call __expf
    // once, then ALL 128 threads reuse the weights → eliminates 127/128 = 99%
    // of __expf calls (was ~8192 per CTA, now 64).
    if (tid < N) {
        float lse_n = smem_w[tid];
        smem_w[tid] = (lse_n == -INFINITY) ? 0.f : __expf(lse_n - lse_max);
    }
    __syncthreads();

    // Phase 4: Compute denominator once (same for all d-elements).
    float den = 0.f;
    for (int n = 0; n < N; ++n) den += smem_w[n];
    float inv_den = (den > 0.f) ? (1.f / den) : 0.f;

    // Phase 5: Weighted sum of partial outputs using precomputed weights.
    // Opart layout: [B, S, NH, N, D] — N×D block for this (batch, m_row, head) is contiguous.
    // Tile-load approach: cooperatively load chunks of Opart into smem for better L2 usage.
    constexpr int kTileN = 64;  // N blocks per smem tile — load entire N dim at once
    // Align sOp_half to 16 bytes for cp.async.cg requirement
    int smem_w_padded = (N + 3) & ~3;  // round up to 4 floats = 16 bytes
    float* sOp = smem_w + smem_w_padded;
    Element* sOp_half = reinterpret_cast<Element*>(sOp);
    // smem layout: sOp_half[kTileN][kHeadDim] in row-major

    float num = 0.f;
    size_t base_o = (((size_t)batch * S + m_row) * NH + head) * N * D;

    for (int n0 = 0; n0 < N; n0 += kTileN) {
        int tile_n = (n0 + kTileN <= N) ? kTileN : (N - n0);

        // Check if entire tile is zero-weight (all masked).
        bool any_nonzero = false;
        for (int tn = 0; tn < tile_n; ++tn) {
            if (smem_w[n0 + tn] != 0.f) { any_nonzero = true; break; }
        }
        if (!any_nonzero) continue;

        // Cooperative load: 128 threads load tile_n × kHeadDim fp16 via cp.async (16B).
        // cp.async bypasses L1 for streaming access patterns.
        {
            constexpr int kLoadElems = 8;  // fp16 per load (16B)
            int total_loads = (tile_n * kHeadDim + kLoadElems - 1) / kLoadElems;
            for (int li = tid; li < total_loads; li += kHeadDim) {
                int elem_idx = li * kLoadElems;
                int tn = elem_idx / kHeadDim;
                int td = elem_idx % kHeadDim;
                if (tn < tile_n) {
                    size_t goff = base_o + (size_t)(n0 + tn) * D + td;
                    Element const* g_src = &Op[goff];
                    Element* s_dst = &sOp_half[tn * kHeadDim + td];
                    uint32_t s_addr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(s_dst));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                                 :: "r"(s_addr), "l"(g_src));
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        // Weighted sum from smem.
        for (int tn = 0; tn < tile_n; ++tn) {
            float w = smem_w[n0 + tn];
            if (w == 0.f) continue;
            num += w * static_cast<float>(sOp_half[tn * kHeadDim + tid]);
        }
        __syncthreads();
    }

    size_t off_final = ((size_t)batch * S + m_row) * NH * D + (size_t)head * D + tid;
    O[off_final] = Element(num * inv_den);
}

} // namespace proj_fused
