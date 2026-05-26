/******************************************************************************
 * SM-Parallel Fused Projection Flash Attention Forward Kernel (SM90, bf16)
 *
 * KV-outer, Q-inner: one CTA per (head, batch), iterates over all KV tiles,
 * then inner loop over Q tiles.  No spinlocks.
 *
 * Grid: (NH, B)  — blockIdx.x=head, blockIdx.y=batch
 * Block: 256 threads = WG0 (0..127) + WG1 (128..255)
 *   WG0: cooperative load helper (all 256 threads participate in loads)
 *   WG1: WGMMA projection + attention computation
 *
 * Phase startup: Load Wk/Wv for this head into persistent smem (sWk, sWv).
 * Outer loop over n_blocks (KV tiles):
 *   Phase 1: Load X[n_block] → sQ (treated as sX)
 *   Phase 1b: WG1 projects sX × sWk → sK, sX × sWv → sV
 *   Phase 2: Inner loop over m_blocks (Q tiles):
 *     Load Q[m_block] → sQ
 *     WG1: QK GEMM → softmax → PV GEMM → LSE-form merge into HBM O_run/LSE_run
 * Finalize: Cast O_run (fp32) → O (bf16)
 ******************************************************************************/
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
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
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>

#include "proj_fused_kernel_traits_sm90.h"
#include "softmax.h"
#include "utils.h"

namespace sm_parallel {

using namespace cute;

// ---------------------------------------------------------------------------
// Barrier enum (CTA-local; IDs do not conflict with other kernels' CTAs)
// ---------------------------------------------------------------------------
enum class SmpBarrier : uint32_t { OSmemReady = 0, MergeStatsReady = 1 };

// ---------------------------------------------------------------------------
// Param struct — fused projection version
// ---------------------------------------------------------------------------
struct SmParallelFusedParams {
    void const* ptr_X;       // [B, S_new, kHiddenDim] bf16  (S_new == seqlen_new)
    void const* ptr_Wk;      // [NH, kHeadDim, kHiddenDim] bf16
    void const* ptr_Wv;      // [NH, kHeadDim, kHiddenDim] bf16
    void const* ptr_Q;       // [B, S_q, NH, kHeadDim] bf16
    void*       ptr_O;       // [B, S_q, NH, kHeadDim] bf16 output
    float*      ptr_O_run;   // [B, S_q, NH, kHeadDim] fp32, init 0
    float*      ptr_LSE_run; // [B, S_q, NH] fp32, init -INF

    int batch, seqlen_new, seqlen_q, num_heads, head_dim, hidden_dim;
    int num_m_blocks, num_n_blocks;
    float softmax_scale;
    int is_causal;
};

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
template <typename Traits>
__global__ __launch_bounds__(256, 1)
void sm_parallel_fused_fwd_kernel(SmParallelFusedParams params)
{
    using Element = typename Traits::Element;
    constexpr int kBlockM        = Traits::kBlockM;
    constexpr int kBlockN        = Traits::kBlockN;
    constexpr int kHeadDim       = Traits::kHeadDim;
    constexpr int kHiddenDim     = Traits::kHiddenDim;
    constexpr int kNThreadsMMA   = Traits::kNThreadsMMA;  // 128
    using SmemLayoutQ  = typename Traits::SmemLayoutQ;
    using SmemLayoutK  = typename Traits::SmemLayoutK;
    using SmemLayoutV  = typename Traits::SmemLayoutV;
    using SmemLayoutO  = typename Traits::SmemLayoutO;
    using TiledMmaProj = typename Traits::TiledMmaProj;
    using TiledMmaQK   = typename Traits::TiledMmaQK;
    using TiledMmaPV   = typename Traits::TiledMmaPV;

    // Full-size W smem layout: [kHeadDim, kHiddenDim], K-major swizzled
    using SmemLayoutAtomW_full = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element,
            cute::Int<Traits::kHeadDim>, cute::Int<Traits::kHiddenDim>>());
    using SmemLayoutW_full = decltype(cute::tile_to_shape(
        SmemLayoutAtomW_full{},
        cute::make_shape(cute::Int<Traits::kHeadDim>{}, cute::Int<Traits::kHiddenDim>{})));

    // For the projection GEMM we need the full X shape [kBlockN, kHiddenDim].
    // Since kBlockN==kBlockM and kHiddenDim==kHeadDim (DefaultTraits), SmemLayoutQ
    // has the same shape/swizzle as what we need for sX.
    // We still use SmemLayoutQ for sQ/sX buffer.

    const int head  = blockIdx.x;
    const int batch = blockIdx.y;
    const int tid   = threadIdx.x;
    const int wg_idx = tid / 128;
    const int ctid   = tid - 128;   // valid only when wg_idx==1

    const int NH       = params.num_heads;
    const int S_q      = params.seqlen_q;
    const int S_new    = params.seqlen_new;

    // Register allocation: WG0 gets few regs, WG1 gets many for MMA fragments
    if (wg_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
    } else {
        cutlass::arch::warpgroup_reg_alloc<240>();
    }

    auto const* X_ptr  = reinterpret_cast<Element const*>(params.ptr_X);
    auto const* Wk_ptr = reinterpret_cast<Element const*>(params.ptr_Wk);
    auto const* Wv_ptr = reinterpret_cast<Element const*>(params.ptr_Wv);
    auto const* Q_ptr  = reinterpret_cast<Element const*>(params.ptr_Q);
    auto*       O_final_ptr = reinterpret_cast<Element*>(params.ptr_O);
    float*      O_run_ptr   = params.ptr_O_run;
    float*      LSE_run_ptr = params.ptr_LSE_run;

    // ------------------------------------------------------------------
    // Shared memory layout (non-aliased):
    //   sQ  [kBlockM, kHeadDim]   SmemLayoutQ        — reused as sX in Phase 1
    //   sWk [kHeadDim, kHiddenDim] SmemLayoutW_full  — persistent
    //   sWv [kHeadDim, kHiddenDim] SmemLayoutW_full  — persistent
    //   sK  [kBlockN, kHeadDim]   SmemLayoutK        — per n_block
    //   sV  [kHeadDim, kBlockN]   SmemLayoutV        — per n_block
    //   sO  [kBlockM, kHeadDim]   SmemLayoutO        — output staging
    //   sLSE_local [kBlockM] float                   — LSE scratch
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto align128 = [](uintptr_t p) -> uintptr_t { return (p + 127) & ~uintptr_t(127); };
    uintptr_t base = align128(reinterpret_cast<uintptr_t>(smem_raw));

    Element* sQ_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));

    Element* sWk_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutW_full{}) * sizeof(Element));

    Element* sWv_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutW_full{}) * sizeof(Element));

    Element* sK_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutK{}) * sizeof(Element));

    Element* sV_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutV{}) * sizeof(Element));

    Element* sO_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutO{}) * sizeof(Element));

    float* sLSE_local = reinterpret_cast<float*>(base);
    // (no further smem allocations needed)

    Tensor sQ  = make_tensor(make_smem_ptr(sQ_ptr),  SmemLayoutQ{});
    Tensor sWk = make_tensor(make_smem_ptr(sWk_ptr), SmemLayoutW_full{});
    Tensor sWv = make_tensor(make_smem_ptr(sWv_ptr), SmemLayoutW_full{});
    Tensor sK  = make_tensor(make_smem_ptr(sK_ptr),  SmemLayoutK{});
    Tensor sV  = make_tensor(make_smem_ptr(sV_ptr),  SmemLayoutV{});
    Tensor sO  = make_tensor(make_smem_ptr(sO_ptr),  SmemLayoutO{});

    // ------------------------------------------------------------------
    // Startup: cooperatively load Wk and Wv for this head into smem.
    // Wk layout: [NH, kHeadDim, kHiddenDim] row-major.
    // Element (d, h) = Wk_ptr + head*kHeadDim*kHiddenDim + d*kHiddenDim + h
    // sWk is swizzled [kHeadDim, kHiddenDim] — must scatter element-by-element.
    // ------------------------------------------------------------------
    {
        constexpr int kEPV    = 8;
        constexpr int kVPR    = kHiddenDim / kEPV;         // uint4 per row
        constexpr int kTotalV = kHeadDim * kVPR;           // total uint4 for one W matrix
        constexpr int kVPT    = (kTotalV + 255) / 256;     // per thread

        auto const* Wk_head = Wk_ptr + (size_t)head * kHeadDim * kHiddenDim;
        auto const* Wv_head = Wv_ptr + (size_t)head * kHeadDim * kHiddenDim;

        #pragma unroll
        for (int v = 0; v < kVPT; ++v) {
            int idx = v * 256 + tid;
            if (idx < kTotalV) {
                int d = idx / kVPR;           // row in [kHeadDim]
                int h = (idx % kVPR) * kEPV;  // col in [kHiddenDim]
                size_t off = (size_t)d * kHiddenDim + h;
                uint4 vk = *reinterpret_cast<uint4 const*>(Wk_head + off);
                uint4 vv = *reinterpret_cast<uint4 const*>(Wv_head + off);
                #pragma unroll
                for (int k = 0; k < kEPV; ++k) {
                    sWk(d, h + k) = reinterpret_cast<Element const*>(&vk)[k];
                    sWv(d, h + k) = reinterpret_cast<Element const*>(&vv)[k];
                }
            }
        }
    }
    __syncthreads();  // sWk, sWv ready

    // ------------------------------------------------------------------
    // Outer loop: for each KV tile (n_block)
    // ------------------------------------------------------------------
    const float softmax_scale_log2 = params.softmax_scale * float(M_LOG2E);
    constexpr int kNRows = 2 * (kBlockM / 64);

    for (int n_block = 0; n_block < params.num_n_blocks; ++n_block) {

        const int n_start = n_block * kBlockN;

        // ----------------------------------------------------------------
        // Phase 1: All 256 threads cooperatively load X[n_block] → sQ
        // X layout: [B, S_new, kHiddenDim] row-major
        // ----------------------------------------------------------------
        {
            constexpr int kEPV = 8;
            constexpr int kVPR = kHiddenDim / kEPV;
            constexpr int kTV  = kBlockN * kVPR;
            constexpr int kVPT = (kTV + 255) / 256;

            size_t base_off = ((size_t)batch * S_new + n_start) * kHiddenDim;
            auto const* X_tile = X_ptr + base_off;

            #pragma unroll
            for (int v = 0; v < kVPT; ++v) {
                int idx = v * 256 + tid;
                if (idx < kTV) {
                    int i = idx / kVPR;
                    int h = (idx % kVPR) * kEPV;
                    size_t row_off = (size_t)i * kHiddenDim + h;
                    uint4 vx = *reinterpret_cast<uint4 const*>(X_tile + row_off);
                    #pragma unroll
                    for (int k = 0; k < kEPV; ++k) {
                        sQ(i, h + k) = reinterpret_cast<Element const*>(&vx)[k];
                    }
                }
            }
        }
        __syncthreads();  // sQ (== sX) ready

        // ----------------------------------------------------------------
        // Phase 1b: WG1 only — projection: sX × sWk → sK, sX × sWv → sV
        // Uses TiledMmaProj (SS-mode [kBlockN, kHeadDim, kHiddenDim])
        // ----------------------------------------------------------------
        if (wg_idx == 1) {
            TiledMmaProj tiled_mma_proj;
            auto thr = tiled_mma_proj.get_thread_slice(ctid);

            // sQ doubles as sX here. For SS projection we need A=[kBlockN,kHiddenDim],
            // B=[kHeadDim,kHiddenDim].
            // TiledMmaProj tile shape is <kProjTileM=64, kHeadDim=128, kHiddenChunk=64>.
            // kBlockN == kProjTileM (both 64) for DefaultTraits.
            // However SmemLayoutQ is [kBlockM, kHeadDim] and sX should be [kBlockN, kHiddenDim].
            // For DefaultTraits kBlockM==kBlockN and kHeadDim==kHiddenDim, so SmemLayoutQ works.

            // For the projection, we need to tile over kHiddenDim in chunks of kHiddenChunk.
            // Since the full W is in smem (size kHeadDim x kHiddenDim), we compute the
            // full projection in one shot if TiledMmaProj can handle it.
            // TiledMmaProj has K-dim = kHiddenChunk (64). We have kHiddenDim = 128 = 2*kHiddenChunk.
            // So we need to loop over kNumProjChunks = 2 iterations.
            // The K-mode of the fragment is kHiddenChunk; we use SmemLayoutQ (full [64,128]) as sX.
            // We need to view sX as [64, 128] and sWk as [128, 128], and iterate K.
            // The TiledMmaProj partition_fragment_A/B will handle the tiling.

            constexpr int kHiddenChunk  = Traits::kHiddenChunk;
            constexpr int kNumProjChunks = Traits::kNumProjChunks;

            // Full-size X layout [kBlockN, kHiddenDim] — same pointer as sQ
            // Full-size W layout [kHeadDim, kHiddenDim] — sWk / sWv
            // We create tensors with the full kHiddenDim size but the tiled_mma
            // has K=kHiddenChunk. partition_fragment_A/B will give fragments
            // with size K = kHiddenDim/kHiddenChunk * kHiddenChunk tiles.
            // Actually for SS WGMMA the fragments are descriptor iterators over smem.
            // The correct approach: use the chunk-wise layout from the Traits
            // but iterate manually.

            // Use SmemLayoutX (chunk-sized [kBlockN, kHiddenChunk]) and iterate
            // We alias into sQ memory using different chunk offsets.
            using SmemLayoutX  = typename Traits::SmemLayoutX;
            using SmemLayoutW  = typename Traits::SmemLayoutW;

            Tensor acc_k = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
            Tensor acc_v = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
            clear(acc_k); clear(acc_v);

            // Iterate over hidden chunks.
            // For swizzled smem layouts, chunk cs starts at cs * cosize(SmemLayoutX{})
            // in the X buffer and cs * cosize(SmemLayoutW{}) in the W buffer.
            // This follows from the K-mode stride in SmemLayoutQ and SmemLayoutW_full:
            // the second 64-column half of [64,128] starts at offset cosize([64,64]).
            constexpr int kChunkStrideX = cute::cosize(SmemLayoutX{});
            constexpr int kChunkStrideW = cute::cosize(SmemLayoutW{});
            #pragma unroll
            for (int cs = 0; cs < kNumProjChunks; ++cs) {
                // Create chunk-sized views
                Element* sX_chunk_ptr  = sQ_ptr  + cs * kChunkStrideX;
                Element* sWk_chunk_ptr = sWk_ptr + cs * kChunkStrideW;
                Element* sWv_chunk_ptr = sWv_ptr + cs * kChunkStrideW;

                Tensor sX_chunk  = make_tensor(make_smem_ptr(sX_chunk_ptr),  SmemLayoutX{});
                Tensor sWk_chunk = make_tensor(make_smem_ptr(sWk_chunk_ptr), SmemLayoutW{});
                Tensor sWv_chunk = make_tensor(make_smem_ptr(sWv_chunk_ptr), SmemLayoutW{});

                Tensor tCsX  = thr.partition_fragment_A(sX_chunk);
                Tensor tCsWk = thr.partition_fragment_B(sWk_chunk);
                Tensor tCsWv = thr.partition_fragment_B(sWv_chunk);

                if (cs == 0) {
                    flash::gemm<true, -1>(tiled_mma_proj, tCsX, tCsWk, acc_k);
                    flash::gemm<true, -1>(tiled_mma_proj, tCsX, tCsWv, acc_v);
                } else {
                    flash::gemm<false, -1>(tiled_mma_proj, tCsX, tCsWk, acc_k);
                    flash::gemm<false, -1>(tiled_mma_proj, tCsX, tCsWv, acc_v);
                }
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(acc_k);
                cute::warpgroup_fence_operand(acc_v);
            }

            // Store acc_k → sK, acc_v → sV (transposed for MN-major sV)
            Tensor tCsK = thr.partition_C(sK);
            Tensor cV   = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
            Tensor tCcV = thr.partition_C(cV);
            #pragma unroll
            for (int i = 0; i < size(acc_k); ++i) {
                tCsK(i) = Element(acc_k(i));
                sV(get<1>(tCcV(i)), get<0>(tCcV(i))) = Element(acc_v(i));
            }
        }
        __syncthreads();  // sK, sV ready for QK GEMM

        // ----------------------------------------------------------------
        // Phase 2: Q-inner loop
        // Causal: Q tile at m_block can attend to this KV tile if
        //   m_start <= n_start + kBlockN - 1
        //   => m_end = min(num_m_blocks, (n_start + kBlockN + kBlockM - 1) / kBlockM)
        // ----------------------------------------------------------------
        const int m_end = params.is_causal
            ? min(params.num_m_blocks, (n_start + kBlockN + kBlockM - 1) / kBlockM)
            : params.num_m_blocks;

        // Set up MMA objects once per n_block (inside WG1 guard below)
        // tOrV is declared per n_block since sV is overwritten each n_block
        // We declare MMA objects for WG1 in a nested scope to keep things tidy

        for (int m_block = 0; m_block < m_end; ++m_block) {
            const int m_start = m_block * kBlockM;

            // ------------------------------------------------------------
            // Step 2a: All 256 threads cooperatively load Q[m_block] → sQ
            // Q layout: [B, S_q, NH, kHeadDim] row-major
            // ------------------------------------------------------------
            {
                constexpr int kEPV = 8;
                constexpr int kVPR = kHeadDim / kEPV;
                constexpr int kTV  = kBlockM * kVPR;
                constexpr int kVPT = (kTV + 255) / 256;

                size_t base_off = ((size_t)batch * S_q + m_start) * NH * kHeadDim
                                + (size_t)head * kHeadDim;
                auto const* Q_tile = Q_ptr + base_off;

                #pragma unroll
                for (int v = 0; v < kVPT; ++v) {
                    int idx = v * 256 + tid;
                    if (idx < kTV) {
                        int i   = idx / kVPR;
                        int d   = (idx % kVPR) * kEPV;
                        size_t row_off = (size_t)i * NH * kHeadDim + d;
                        uint4 vq = *reinterpret_cast<uint4 const*>(Q_tile + row_off);
                        #pragma unroll
                        for (int k = 0; k < kEPV; ++k) {
                            sQ(i, d + k) = reinterpret_cast<Element const*>(&vq)[k];
                        }
                    }
                }
            }
            __syncthreads();  // sQ ready

            // ------------------------------------------------------------
            // Step 2b: WG1 only — QK GEMM + softmax + PV GEMM + merge
            // ------------------------------------------------------------
            if (wg_idx == 1) {
                TiledMmaQK tiled_mma_qk;
                TiledMmaPV tiled_mma_pv;
                auto thr_mma_qk = tiled_mma_qk.get_thread_slice(ctid);
                auto thr_mma_pv = tiled_mma_pv.get_thread_slice(ctid);

                Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
                Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
                Tensor tOrV = thr_mma_pv.partition_fragment_B(sV);

                // QK GEMM
                Tensor acc_s = partition_fragment_C(tiled_mma_qk,
                                                    Shape<Int<kBlockM>, Int<kBlockN>>{});
                flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK, acc_s);
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(acc_s);

                // Causal mask
                if (params.is_causal) {
                    auto thread0_mma = TiledMmaQK{}.get_thread_slice(_0{});
                    Tensor cS     = cute::make_identity_tensor(
                                        Shape<Int<kBlockM>, Int<kBlockN>>{});
                    Tensor tScS   = thr_mma_qk.partition_C(cS);
                    Tensor t0ScS  = thread0_mma.partition_C(cS);
                    Tensor acc_s_rc = make_tensor(acc_s.data(),
                        flash::convert_layout_acc_rowcol(acc_s.layout()));
                    Tensor tScS_rc  = make_tensor(tScS.data(),
                        flash::convert_layout_acc_rowcol(tScS.layout()));
                    Tensor t0ScS_rc = make_tensor(t0ScS.data(),
                        flash::convert_layout_acc_rowcol(t0ScS.layout()));

                    int thread_col_offset = get<1>(tScS_rc(_0{}, _0{}));
                    // Q token at m_start + row_rel can attend to KV tokens
                    // up to position (m_start + row_rel) inclusive (causal).
                    // KV global position = n_start + thread_col_offset + col_rel_t0
                    // Mask where: n_start + thread_col_offset + col_rel_t0 > m_start + row_rel
                    int causal_row_offset = m_start + 1 - n_start - thread_col_offset;
                    #pragma unroll
                    for (int m = 0; m < size<0>(acc_s_rc); ++m) {
                        int row_rel = get<0>(tScS_rc(m, _0{}));
                        int col_limit = row_rel + causal_row_offset;
                        #pragma unroll
                        for (int n = 0; n < size<1>(acc_s_rc); ++n) {
                            int col_rel_t0 = get<1>(t0ScS_rc(_0{}, n));
                            if (col_rel_t0 >= col_limit) {
                                acc_s_rc(m, n) = -INFINITY;
                            }
                        }
                    }
                }

                // Softmax (Is_first=true: single KV tile per Q block, standalone)
                flash::Softmax<kNRows, 0> softmax(softmax_scale_log2);
                (void)softmax.template max_get_scale<true, true>(acc_s);
                softmax.template online_softmax<true, true>(acc_s);

                // Convert acc_s (fp32 softmax output) → tOrP (bf16)
                Tensor tOrP_acc = make_tensor(acc_s.data(),
                    flash::convert_layout_acc_Aregs<TiledMmaPV>(acc_s.layout()));
                Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
                flash::convert_type_out(tOrP_acc, tOrP);

                // PV GEMM
                Tensor acc_o_local = partition_fragment_C(tiled_mma_pv,
                                                          Shape<Int<kBlockM>, Int<kHeadDim>>{});
                flash::gemm<true, -1>(tiled_mma_pv, tOrP, tOrV, acc_o_local);

                // Finalize softmax: reduce row_sum, compute LSE, get scale = 1/l_local
                auto scores_scale = softmax.finalize();
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(acc_o_local);

                // Normalize acc_o_local by 1/l_local
                softmax.rescale_o(acc_o_local, scores_scale);

                // (a) Write LSE to sLSE_local (row-leader pattern: lane % 4 == 0)
                {
                    Tensor cO   = cute::make_identity_tensor(
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{});
                    Tensor tOcO = thr_mma_pv.partition_C(cO);
                    int lane = ctid % 32;
                    if (lane % 4 == 0) {
                        Tensor tOcO_rc = make_tensor(tOcO.data(),
                            flash::convert_layout_acc_rowcol(tOcO.layout()));
                        #pragma unroll
                        for (int mi = 0; mi < kNRows; ++mi) {
                            int row_rel = get<0>(tOcO_rc(mi, _0{}));
                            sLSE_local[row_rel] = softmax.row_sum(mi);
                        }
                    }
                }

                // (b) STSM acc_o_local → sO
                {
                    auto smem_tiled_copy_O = make_tiled_copy_C(
                        cute::Copy_Atom<cute::SM90_U32x4_STSM_N, Element>{}, tiled_mma_pv);
                    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(ctid);
                    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

                    Tensor rO = make_tensor_like<Element>(acc_o_local);
                    flash::convert_type_out(acc_o_local, rO);
                    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
                    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
                }

                // Ensure sO and sLSE_local are visible to all 128 MMA threads
                cutlass::arch::NamedBarrier::sync(
                    static_cast<uint32_t>(kNThreadsMMA),
                    static_cast<uint32_t>(SmpBarrier::OSmemReady));

                // (c) Lock-free merge: LSE-form online merge into HBM O_run/LSE_run
                // Threading: 2 threads/row × kBlockM rows = 2*kBlockM = 128 threads
                {
                    int row_in_tile = ctid >> 1;      // 0..63 (kBlockM-1)
                    int d_half      = ctid & 1;
                    int s_global    = m_start + row_in_tile;

                    if (s_global < S_q) {
                        int row_id = (batch * S_q + s_global) * NH + head;

                        float LSE_local_val = sLSE_local[row_in_tile];
                        float LSE_old = LSE_run_ptr[row_id];

                        // LSE-form merge
                        float alpha, beta, LSE_new;
                        float LSE_max = fmaxf(LSE_old, LSE_local_val);
                        if (LSE_max == -INFINITY) {
                            alpha = 0.f; beta = 0.f; LSE_new = -INFINITY;
                        } else {
                            float a_old = (LSE_old       == -INFINITY) ? 0.f
                                : __expf(LSE_old       - LSE_max);
                            float a_loc = (LSE_local_val == -INFINITY) ? 0.f
                                : __expf(LSE_local_val - LSE_max);
                            float Z = a_old + a_loc;
                            LSE_new = LSE_max + __logf(Z);
                            alpha = (LSE_old       == -INFINITY) ? 0.f
                                : __expf(LSE_old       - LSE_new);
                            beta  = (LSE_local_val == -INFINITY) ? 0.f
                                : __expf(LSE_local_val - LSE_new);
                        }

                        // Each thread updates kHeadDim/2 elements
                        constexpr int kHalfD = kHeadDim / 2;
                        int d_start = d_half * kHalfD;
                        #pragma unroll
                        for (int d = 0; d < kHalfD; d += 4) {
                            int d_global = d_start + d;
                            float4 o_old = *reinterpret_cast<const float4*>(
                                &O_run_ptr[row_id * kHeadDim + d_global]);
                            Element so0_ = sO(row_in_tile, d_global + 0);
                            Element so1_ = sO(row_in_tile, d_global + 1);
                            Element so2_ = sO(row_in_tile, d_global + 2);
                            Element so3_ = sO(row_in_tile, d_global + 3);
                            float4 o_new;
                            o_new.x = alpha * o_old.x + beta * (float)so0_;
                            o_new.y = alpha * o_old.y + beta * (float)so1_;
                            o_new.z = alpha * o_old.z + beta * (float)so2_;
                            o_new.w = alpha * o_old.w + beta * (float)so3_;
                            *reinterpret_cast<float4*>(
                                &O_run_ptr[row_id * kHeadDim + d_global]) = o_new;
                        }

                        // Only one thread per row writes the new LSE
                        if (d_half == 0) {
                            LSE_run_ptr[row_id] = LSE_new;
                        }
                    }

                    cutlass::arch::NamedBarrier::sync(
                        static_cast<uint32_t>(kNThreadsMMA),
                        static_cast<uint32_t>(SmpBarrier::MergeStatsReady));
                }
            }  // end if (wg_idx == 1)

            __syncthreads();  // guard sQ reuse for next m_block
        }  // end m_block loop

        // sV is about to be overwritten by the next n_block's projection —
        // no additional sync needed here since __syncthreads() at end of m_block loop
        // (or the initial __syncthreads after projection) covers it.
    }  // end n_block loop

    // ------------------------------------------------------------------
    // Finalize: cast O_run (fp32) → O (bf16) and write final output
    // This CTA owns the entire (batch, head) slice: seqlen_q * kHeadDim elements
    // ------------------------------------------------------------------
    __syncthreads();  // ensure all merges are visible

    {
        int total = S_q * kHeadDim;
        for (int idx = tid; idx < total; idx += 256) {
            int s = idx / kHeadDim;
            int d = idx % kHeadDim;
            int row_id = (batch * S_q + s) * NH + head;
            float o_val = O_run_ptr[row_id * kHeadDim + d];
            size_t off  = (((size_t)batch * S_q + s) * NH + head) * kHeadDim + d;
            O_final_ptr[off] = (Element)o_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Smem size helper
// ---------------------------------------------------------------------------
template <typename Traits>
size_t sm_parallel_fused_smem_bytes()
{
    using Element = typename Traits::Element;

    using SmemLayoutAtomW_full = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element,
            cute::Int<Traits::kHeadDim>, cute::Int<Traits::kHiddenDim>>());
    using SmemLayoutW_full = decltype(cute::tile_to_shape(
        SmemLayoutAtomW_full{},
        cute::make_shape(cute::Int<Traits::kHeadDim>{}, cute::Int<Traits::kHiddenDim>{})));

    auto r128 = [](size_t s) { return (s + 127) & ~size_t(127); };
    size_t smem = 128;
    smem += r128(cute::cosize(typename Traits::SmemLayoutQ{}) * sizeof(Element));   // sQ
    smem += r128(cute::cosize(SmemLayoutW_full{}) * sizeof(Element));               // sWk
    smem += r128(cute::cosize(SmemLayoutW_full{}) * sizeof(Element));               // sWv
    smem += r128(cute::cosize(typename Traits::SmemLayoutK{}) * sizeof(Element));   // sK
    smem += r128(cute::cosize(typename Traits::SmemLayoutV{}) * sizeof(Element));   // sV
    smem += r128(cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));   // sO
    smem += Traits::kBlockM * sizeof(float) + 16;                                   // sLSE_local
    return smem;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
template <typename Traits>
cudaError_t launch_sm_parallel_fused(SmParallelFusedParams p, cudaStream_t stream = 0)
{
    size_t smem = sm_parallel_fused_smem_bytes<Traits>();

    auto* kernel = &sm_parallel_fused_fwd_kernel<Traits>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (err != cudaSuccess) return err;

    dim3 grid(p.num_heads, p.batch);
    dim3 block(256);
    kernel<<<grid, block, smem, stream>>>(p);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Legacy pre-projected interface (Grid=(num_m_blocks, NH, B), no projection)
// ---------------------------------------------------------------------------
struct SmParallelParams {
    void const* __restrict__ ptr_K;   // [B, S_kv, NH, D] bf16
    void const* __restrict__ ptr_V;   // [B, S_kv, NH, D] bf16
    void const* __restrict__ ptr_Q;   // [B, S_q,  NH, D] bf16
    void*       __restrict__ ptr_O;   // [B, S_q,  NH, D] bf16 (output)

    int batch, seqlen_q, seqlen_kv;
    int num_heads, head_dim;
    int num_m_blocks, num_n_blocks;
    int seqlen_past;
    float softmax_scale;
    int is_causal;
};

template <typename Traits>
__global__ __launch_bounds__(256, 1)
void sm_parallel_fwd_kernel(SmParallelParams params)
{
    using Element = typename Traits::Element;
    constexpr int kBlockM      = Traits::kBlockM;
    constexpr int kBlockN      = Traits::kBlockN;
    constexpr int kHeadDim     = Traits::kHeadDim;
    constexpr int kNThreadsMMA = Traits::kNThreadsMMA;
    using SmemLayoutQ = typename Traits::SmemLayoutQ;
    using SmemLayoutK = typename Traits::SmemLayoutK;
    using SmemLayoutV = typename Traits::SmemLayoutV;
    using SmemLayoutO = typename Traits::SmemLayoutO;
    using TiledMmaQK  = typename Traits::TiledMmaQK;
    using TiledMmaPV  = typename Traits::TiledMmaPV;

    const int m_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int tid     = threadIdx.x;
    const int wg_idx  = tid / 128;
    const int ctid    = (wg_idx == 1) ? (tid - 128) : 0;

    const int S_q    = params.seqlen_q;
    const int S_kv   = params.seqlen_kv;
    const int NH     = params.num_heads;
    const int S_past = params.seqlen_past;
    const int m_start = m_block * kBlockM;

    auto const* K_ptr = reinterpret_cast<Element const*>(params.ptr_K);
    auto const* V_ptr = reinterpret_cast<Element const*>(params.ptr_V);
    auto const* Q_ptr = reinterpret_cast<Element const*>(params.ptr_Q);
    auto*       O_ptr = reinterpret_cast<Element*>(params.ptr_O);

    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto align128 = [](uintptr_t p) -> uintptr_t { return (p + 127) & ~uintptr_t(127); };
    uintptr_t base = align128(reinterpret_cast<uintptr_t>(smem_raw));

    Element* sQ_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));
    Element* sK_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutK{}) * sizeof(Element));
    Element* sV_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutV{}) * sizeof(Element));
    Element* sO_ptr = reinterpret_cast<Element*>(base);

    Tensor sQ = make_tensor(make_smem_ptr(sQ_ptr), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(sK_ptr), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutV{});
    Tensor sO = make_tensor(make_smem_ptr(sO_ptr), SmemLayoutO{});

    // Load Q
    {
        constexpr int kEPV = 8, kVPR = kHeadDim / kEPV;
        constexpr int kTV = kBlockM * kVPR, kVPT = (kTV + 255) / 256;
        size_t base_off = ((size_t)batch * S_q + m_start) * NH * kHeadDim + (size_t)head * kHeadDim;
        auto const* Q_tile = Q_ptr + base_off;
        #pragma unroll
        for (int v = 0; v < kVPT; ++v) {
            int idx = v * 256 + tid;
            if (idx < kTV) {
                int i = idx / kVPR, d = (idx % kVPR) * kEPV;
                uint4 vq = *reinterpret_cast<uint4 const*>(Q_tile + (size_t)i * NH * kHeadDim + d);
                #pragma unroll
                for (int k = 0; k < kEPV; ++k) sQ(i, d + k) = reinterpret_cast<Element const*>(&vq)[k];
            }
        }
    }
    __syncthreads();

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(ctid);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(ctid);
    Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
    Tensor tOrV = thr_mma_pv.partition_fragment_B(sV);
    Tensor acc_o = partition_fragment_C(tiled_mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    auto smem_tiled_copy_O = make_tiled_copy_C(cute::Copy_Atom<cute::SM90_U32x4_STSM_N, Element>{}, tiled_mma_pv);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(ctid);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    constexpr int kNRows = 2 * (kBlockM / 64);
    float const softmax_scale_log2 = params.softmax_scale * float(M_LOG2E);
    flash::Softmax<kNRows, 0> softmax(softmax_scale_log2);
    if (wg_idx == 1) { clear(acc_o); }

    const int n_end = params.is_causal
        ? (S_past + m_start + kBlockM + kBlockN - 1) / kBlockN
        : params.num_n_blocks;

    for (int n_block = 0; n_block < n_end; ++n_block) {
        {
            constexpr int kEPV = 8, kVPR = kHeadDim / kEPV;
            constexpr int kTV = kBlockN * kVPR, kVPT = (kTV + 255) / 256;
            int n_start = n_block * kBlockN;
            size_t base_off = ((size_t)batch * S_kv + n_start) * NH * kHeadDim + (size_t)head * kHeadDim;
            auto const* K_tile = K_ptr + base_off;
            auto const* V_tile = V_ptr + base_off;
            #pragma unroll
            for (int v = 0; v < kVPT; ++v) {
                int idx = v * 256 + tid;
                if (idx < kTV) {
                    int i = idx / kVPR, d = (idx % kVPR) * kEPV;
                    uint4 vk = *reinterpret_cast<uint4 const*>(K_tile + (size_t)i * NH * kHeadDim + d);
                    uint4 vv = *reinterpret_cast<uint4 const*>(V_tile + (size_t)i * NH * kHeadDim + d);
                    #pragma unroll
                    for (int k = 0; k < kEPV; ++k) {
                        sK(i, d + k) = reinterpret_cast<Element const*>(&vk)[k];
                        sV(d + k, i) = reinterpret_cast<Element const*>(&vv)[k];
                    }
                }
            }
        }
        __syncthreads();

        if (wg_idx == 1) {
            Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
            flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK, acc_s);
            cute::warpgroup_wait<0>(); cute::warpgroup_fence_operand(acc_s);

            if (params.is_causal) {
                auto thread0_mma = TiledMmaQK{}.get_thread_slice(_0{});
                Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
                Tensor tScS = thr_mma_qk.partition_C(cS), t0ScS = thread0_mma.partition_C(cS);
                Tensor ar = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                Tensor tr = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol(tScS.layout()));
                Tensor t0r = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol(t0ScS.layout()));
                int n_start = n_block * kBlockN;
                int tc = get<1>(tr(_0{}, _0{}));
                int cr = S_past + m_start + 1 - n_start - tc;
                #pragma unroll
                for (int m = 0; m < size<0>(ar); ++m) {
                    int cl = get<0>(tr(m, _0{})) + cr;
                    #pragma unroll
                    for (int n = 0; n < size<1>(ar); ++n)
                        if (get<1>(t0r(_0{}, n)) >= cl) ar(m, n) = -INFINITY;
                }
            }

            typename flash::Softmax<kNRows, 0>::TensorT scores_scale;
            if (n_block == 0) {
                scores_scale = softmax.template max_get_scale<true, true>(acc_s);
            } else {
                scores_scale = softmax.template max_get_scale<false, true>(acc_s);
                softmax.rescale_o(acc_o, scores_scale);
            }
            if (n_block == 0) softmax.template online_softmax<true, true>(acc_s);
            else               softmax.template online_softmax<false, true>(acc_s);

            Tensor tOrP_acc = make_tensor(acc_s.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(acc_s.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            flash::convert_type_out(tOrP_acc, tOrP);
            if (n_block == 0) flash::gemm<true, 0>(tiled_mma_pv, tOrP, tOrV, acc_o);
            else               flash::gemm<false, 0>(tiled_mma_pv, tOrP, tOrV, acc_o);
            cute::warpgroup_fence_operand(acc_o);
        }
        __syncthreads();
    }

    if (wg_idx == 1) {
        auto scores_scale = softmax.finalize();
        softmax.rescale_o(acc_o, scores_scale);
        Tensor rO = make_tensor_like<Element>(acc_o);
        flash::convert_type_out(acc_o, rO);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    }
    __syncthreads();

    {
        constexpr int kEPV = 8, kVPR = kHeadDim / kEPV;
        constexpr int kTV = kBlockM * kVPR, kVPT = (kTV + 255) / 256;
        #pragma unroll
        for (int v = 0; v < kVPT; ++v) {
            int idx = v * 256 + tid;
            if (idx < kTV) {
                int i = idx / kVPR, d = (idx % kVPR) * kEPV;
                int s_global = m_start + i;
                if (s_global < S_q) {
                    size_t off = ((size_t)batch * S_q + s_global) * NH * kHeadDim + (size_t)head * kHeadDim + d;
                    uint4 vo;
                    #pragma unroll
                    for (int k = 0; k < kEPV; ++k) reinterpret_cast<Element*>(&vo)[k] = sO(i, d + k);
                    *reinterpret_cast<uint4*>(O_ptr + off) = vo;
                }
            }
        }
    }
}

template <typename Traits>
cudaError_t launch_sm_parallel(SmParallelParams p, cudaStream_t stream = 0)
{
    using Element = typename Traits::Element;
    auto r128 = [](size_t s) { return (s + 127) & ~size_t(127); };
    size_t smem = 128;
    smem += r128(cute::cosize(typename Traits::SmemLayoutQ{}) * sizeof(Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutK{}) * sizeof(Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutV{}) * sizeof(Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));
    p.num_m_blocks = (p.seqlen_q  + Traits::kBlockM - 1) / Traits::kBlockM;
    p.num_n_blocks = (p.seqlen_kv + Traits::kBlockN - 1) / Traits::kBlockN;
    if (p.seqlen_past < 0) p.seqlen_past = 0;
    dim3 grid(p.num_m_blocks, p.num_heads, p.batch);
    dim3 block(256);
    auto* kernel = &sm_parallel_fwd_kernel<Traits>;
    cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (err != cudaSuccess) return err;
    kernel<<<grid, block, smem, stream>>>(p);
    return cudaGetLastError();
}

} // namespace sm_parallel
