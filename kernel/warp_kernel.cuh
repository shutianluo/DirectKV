/******************************************************************************
 * Projection-Fused Flash Attention — FA3-style Warp-Specialized Kernel (SM90)
 *
 * Best FA3-style version: 858µs @ S=4096 (vs 712µs baseline kernel.cuh).
 * Slower due to occupancy 1 (256 threads, launch_bounds(256,1)) vs baseline
 * occupancy 2 (128 threads, launch_bounds(128,2)).
 *
 * Architecture:
 *   WG0 (threads 0-127): Producer — issues TMA Q loads, spins on smem_done
 *                         flags waiting for consumer to release Q buffers.
 *   WG1 (threads 128-255): Consumer — projection + attention compute.
 *
 * Phase 1 (projection): ALL 256 threads participate in cp.async loads.
 *   Only WG1 does WGMMA. __syncthreads() between phases ensures sX=sQ0
 *   alias safety (sX is overwritten by TMA in phase 2).
 *
 * Phase 2 (attention): Producer loads Q tiles via TMA into double-buffered
 *   sQ0/sQ1. Consumer does QK GEMM, softmax, PV GEMM, Opart stores.
 *   Synchronization: mbarrier for TMA completion (producer→consumer),
 *   smem_done[buf] for buffer release (consumer→producer).
 *
 * Grid: (num_n_blocks, num_heads, batch) — KV-centric layout.
 *
 * To use: change c_api.cu to include this file and launch with 256 threads:
 *   #include "warp_kernel.cuh"
 *   dim3 block(256);
 *   proj_fused::proj_fused_fwd_kernel_warpspec<Traits><<<grid, block, smem, 0>>>(p);
 *
 * Requires: -arch=compute_90a -code=sm_90a (for mbarrier TMA instructions).
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

////////////////////////////////////////////////////////////////////////////////
// FA3-style warp-specialized kernel (2 warpgroups: producer + consumer)
////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
__global__ __launch_bounds__(256, 1)
void proj_fused_fwd_kernel_warpspec(ProjFusedParams params)
{
    using Element = typename Traits::Element;
    constexpr int kBlockM   = Traits::kBlockM;
    constexpr int kBlockN   = Traits::kBlockN;
    constexpr int kHeadDim  = Traits::kHeadDim;
    constexpr int kHiddenChunk = Traits::kHiddenChunk;
    constexpr int kNumProjChunks = Traits::kNumProjChunks;
    constexpr int kNThreadsMMA = Traits::kNThreadsMMA;  // 128
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

    // Warpgroup assignment
    const int wg_idx = tid / 128;       // 0 = producer, 1 = consumer
    const int ctid   = tid - 128;       // consumer-local tid (0-127), only valid for WG1

    const int S = params.seqlen;
    const int H = params.hidden_dim;
    const int D = params.head_dim;
    const int NH = params.num_heads;

    const int n_start = n_block * kBlockN;
    const int S_past = params.seqlen_past;
    const int num_n_blocks_past = params.num_n_blocks_past;
    const bool is_past_kv = (n_block < num_n_blocks_past);
    const int n_start_new = n_start - S_past;

    auto const* X_ptr  = reinterpret_cast<Element const*>(params.ptr_X);
    auto const* Wk_ptr = reinterpret_cast<Element const*>(params.ptr_Wk);
    auto const* Wv_ptr = reinterpret_cast<Element const*>(params.ptr_Wv);
    auto* Op_ptr = reinterpret_cast<Element*>(params.ptr_Opart);
    auto* Kc_ptr = reinterpret_cast<Element*>(params.ptr_Kc);
    auto* Vc_ptr = reinterpret_cast<Element*>(params.ptr_Vc);
    float* LSE_ptr = params.ptr_LSE;
    auto const* Kpast_ptr = reinterpret_cast<Element const*>(params.ptr_Kpast);
    auto const* Vpast_ptr = reinterpret_cast<Element const*>(params.ptr_Vpast);

    Element const* gX  = is_past_kv ? nullptr
                                    : X_ptr + (size_t)batch * S * H + (size_t)n_start_new * H;
    Element const* gWk = Wk_ptr + (size_t)head * D * H;
    Element const* gWv = Wv_ptr + (size_t)head * D * H;

    // ------------------------------------------------------------------
    // Shared memory layout (aliased): sQ0 | sQ1 | sK | sV | sO0/sWk | sO1/sWv
    // sX aliases sQ0 (used only during projection phase).
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto align128 = [](uintptr_t p) -> uintptr_t { return (p + 127) & ~uintptr_t(127); };
    uintptr_t base = reinterpret_cast<uintptr_t>(smem_raw);
    base = align128(base);
    Element* sQ0_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));
    Element* sQ1_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));
    Element* sK_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutK{}) * sizeof(Element));
    Element* sV_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutV{}) * sizeof(Element));
    Element* sO0_ptr = reinterpret_cast<Element*>(base);
    Element* sWk_ptr = sO0_ptr;
    base = align128(base + cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));
    Element* sO1_ptr = reinterpret_cast<Element*>(base);
    Element* sWv_ptr = sO1_ptr;
    Element* sX_ptr = sQ0_ptr;  // sX aliases sQ0

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
    // mbarrier for TMA Q completion + consumer-done flags
    // ------------------------------------------------------------------
    constexpr int kTmaHalfBytes = kBlockM * 64 * (int)sizeof(Element);

    // mbar_Q[2]: mbarrier for TMA → consumer signaling (placed after sO1)
    uint64_t* mbar_Q = reinterpret_cast<uint64_t*>(
        reinterpret_cast<uintptr_t>(sO1_ptr) +
        cute::cosize(SmemLayoutO{}) * sizeof(Element));
    mbar_Q = reinterpret_cast<uint64_t*>((reinterpret_cast<uintptr_t>(mbar_Q) + 7) & ~uintptr_t(7));

    // smem_done[2]: consumer→producer signaling flags (after mbar_Q[2])
    int* smem_done = reinterpret_cast<int*>(mbar_Q + 2);

    if (tid == 0) {
        uint32_t mb0 = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_Q[0]));
        uint32_t mb1 = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_Q[1]));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mb0), "r"(1));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mb1), "r"(1));
        smem_done[0] = 1;  // both buffers start as "available"
        smem_done[1] = 1;
    }
    __syncthreads();

    auto bar_consumer = [](){ asm volatile("bar.sync 1, 128;\n" ::); };

    const int new_n_idx = n_block - num_n_blocks_past;
    const int m_start_block = (params.is_causal && new_n_idx > 0) ? new_n_idx : 0;

    // ==================================================================
    // PHASE 1: K/V Projection — ALL 256 threads participate in loads,
    // only consumer WG1 does WGMMA. sX aliases sQ0.
    // ==================================================================
    if (is_past_kv) {
        constexpr int kElemsPerVec = 8;
        constexpr int kVecsPerRow = kHeadDim / kElemsPerVec;
        constexpr int kTotalVecs = kBlockN * kVecsPerRow;
        constexpr int kVecsPerThread256 = (kTotalVecs + 255) / 256;
        #pragma unroll
        for (int v = 0; v < kVecsPerThread256; ++v) {
            int idx = v * 256 + tid;
            if (idx < kTotalVecs) {
                int i = idx / kVecsPerRow;
                int d = (idx % kVecsPerRow) * kElemsPerVec;
                int row = n_start + i;
                size_t off = ((size_t)batch * S_past + row) * NH * D + (size_t)head * D + d;
                uint4 vk = *reinterpret_cast<uint4 const*>(Kpast_ptr + off);
                uint4 vv = *reinterpret_cast<uint4 const*>(Vpast_ptr + off);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    sK(i, d + k) = reinterpret_cast<Element const*>(&vk)[k];
                    sV(d + k, i) = reinterpret_cast<Element const*>(&vv)[k];
                }
            }
        }
    } else {
        TiledMmaProj tiled_mma_proj;
        auto thr_mma = tiled_mma_proj.get_thread_slice(wg_idx == 1 ? ctid : 0);

        Tensor acc_k = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor acc_v = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
        if (wg_idx == 1) { clear(acc_k); clear(acc_v); }

        Tensor tCsX  = thr_mma.partition_fragment_A(sX);
        Tensor tCsWk = thr_mma.partition_fragment_B(sWk);
        Tensor tCsWv = thr_mma.partition_fragment_B(sWv);

        constexpr int kElemsPerVec = 8;

        auto load_X_all = [&](Element const* gSrc, int cs) {
            constexpr int kVPR = kHiddenChunk / kElemsPerVec;
            constexpr int kTV = kBlockN * kVPR;
            constexpr int kVPT = (kTV + 255) / 256;
            #pragma unroll
            for (int v = 0; v < kVPT; ++v) {
                int idx = v * 256 + tid;
                if (idx < kTV) {
                    int row = idx / kVPR, col = (idx % kVPR) * kElemsPerVec;
                    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(&sX(row, col)));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                        :: "r"(sa), "l"(gSrc + (size_t)row * H + cs * kHiddenChunk + col));
                }
            }
        };
        auto load_W_all = [&](Element const* gSrc, int cs, auto& sDst) {
            constexpr int kVPR = kHiddenChunk / kElemsPerVec;
            constexpr int kTV = kHeadDim * kVPR;
            constexpr int kVPT = (kTV + 255) / 256;
            #pragma unroll
            for (int v = 0; v < kVPT; ++v) {
                int idx = v * 256 + tid;
                if (idx < kTV) {
                    int row = idx / kVPR, col = (idx % kVPR) * kElemsPerVec;
                    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(&sDst(row, col)));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                        :: "r"(sa), "l"(gSrc + (size_t)row * H + cs * kHiddenChunk + col));
                }
            }
        };

        #pragma unroll
        for (int cs = 0; cs < kNumProjChunks; ++cs) {
            load_X_all(gX, cs);
            load_W_all(gWk, cs, sWk);
            load_W_all(gWv, cs, sWv);
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

        if (wg_idx == 1) {
            Tensor tCsK = thr_mma.partition_C(sK);
            Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
            Tensor tCcV = thr_mma.partition_C(cV);
            #pragma unroll
            for (int i = 0; i < size(acc_k); ++i) {
                tCsK(i) = Element(acc_k(i));
                sV(get<1>(tCcV(i)), get<0>(tCcV(i))) = Element(acc_v(i));
            }
        }
    }
    __syncthreads();  // sX=sQ0 now free for TMA

    // KV cache store (all 256 threads)
    if (!is_past_kv) {
        constexpr int kEPV = 8;
        constexpr int kVPR = kHeadDim / kEPV;
        constexpr int kTV = kBlockN * kVPR;
        constexpr int kVPT = (kTV + 255) / 256;
        #pragma unroll
        for (int v = 0; v < kVPT; ++v) {
            int idx = v * 256 + tid;
            if (idx < kTV) {
                int i = idx / kVPR, d = (idx % kVPR) * kEPV;
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
    }

    // ==================================================================
    // PHASE 2: Producer/Consumer diverge for attention loop
    // ==================================================================

    // ---- PRODUCER (WG0): TMA Q loader ----
    if (wg_idx == 0) {
        CUtensorMap const* tma_desc = params.tma_desc_Q;

        auto producer_load_Q = [&](int m_start, int buf) {
            Element* sq = (buf == 0) ? sQ0_ptr : sQ1_ptr;
            uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(sq));
            uint32_t s1 = s0 + kTmaHalfBytes;
            uint32_t mb = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_Q[buf]));
            if (tid == 0) {
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                             :: "r"(mb), "r"(2 * kTmaHalfBytes));
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
                    :: "r"(s0), "l"(tma_desc),
                       "r"(0), "r"(head), "r"(m_start), "r"(batch), "r"(mb));
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
                    :: "r"(s1), "l"(tma_desc),
                       "r"(64), "r"(head), "r"(m_start), "r"(batch), "r"(mb));
                smem_done[buf] = 0;  // mark buffer in-use
            }
        };

        // Load Q tiles, waiting for consumer "done" signal before buffer reuse
        for (int m_block = m_start_block; m_block < params.num_m_blocks; ++m_block) {
            int buf = m_block & 1;
            if (tid == 0) {
                while (atomicAdd(&smem_done[buf], 0) == 0) { __nanosleep(32); }
            }
            __syncwarp(0xffffffff);
            producer_load_Q(m_block * kBlockM, buf);
        }
        return;  // producer done
    }

    // ---- CONSUMER (WG1): attention compute ----
    float const softmax_scale_log2 = params.softmax_scale * float(M_LOG2E);
    constexpr int kNRows = 2 * (kBlockM / 64);
    using SoftmaxT = flash::Softmax<kNRows, /*Max_offset=*/0>;

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(ctid);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(ctid);

    Tensor tSrQ0 = thr_mma_qk.partition_fragment_A(sQ0);
    Tensor tSrQ1 = thr_mma_qk.partition_fragment_A(sQ1);
    Tensor tSrK  = thr_mma_qk.partition_fragment_B(sK);
    Tensor tOrV  = thr_mma_pv.partition_fragment_B(sV);

    auto smem_tiled_copy_O = make_tiled_copy_C(
        cute::Copy_Atom<cute::SM90_U32x4_STSM_N, Element>{}, tiled_mma_pv);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(ctid);
    Tensor taccOsO0 = smem_thr_copy_O.partition_D(sO0);
    Tensor taccOsO1 = smem_thr_copy_O.partition_D(sO1);
    const int num_n_blocks_o = params.num_n_blocks;
    constexpr int kElemsPerVecO = 8;
    constexpr int kVecsPerRowO = kHeadDim / kElemsPerVecO;
    constexpr int kTotalVecsO  = kBlockM * kVecsPerRowO;
    constexpr int kVecsPerThreadO = kTotalVecsO / kNThreadsMMA;

    // TMA wait: consumer waits for producer's TMA to deliver Q data
    int tma_phase[2] = {0, 0};
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

    // Signal producer that consumer finished reading Q from buffer
    auto consumer_signal_done = [&](int buf) {
        bar_consumer();
        if (ctid == 0) { atomicExch(&smem_done[buf], 1); }
    };

    // Write LSE=-INF for skipped m_blocks (causal)
    {
        const int num_n_blocks_lse = params.num_n_blocks;
        for (int m_block = 0; m_block < m_start_block; ++m_block) {
            int m_start = m_block * kBlockM;
            for (int i = ctid; i < kBlockM; i += kNThreadsMMA) {
                int row = m_start + i;
                size_t off_lse = (((size_t)batch * S + row) * NH + head) * num_n_blocks_lse + n_block;
                LSE_ptr[off_lse] = -INFINITY;
            }
        }
    }

    // Main attention loop
    for (int m_block = m_start_block; m_block < params.num_m_blocks; ++m_block) {
        int m_start = m_block * kBlockM;
        int buf = m_block & 1;

        // Wait for producer's TMA to complete
        tma_wait_Q(buf);
        bar_consumer();

        // GEMM-I: S = Q · K^T
        Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
        if (buf == 0) {
            flash::gemm<true, -1>(tiled_mma_qk, tSrQ0, tSrK, acc_s);
        } else {
            flash::gemm<true, -1>(tiled_mma_qk, tSrQ1, tSrK, acc_s);
        }
        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(acc_s);

        // Q data consumed — release buffer for producer to reuse
        consumer_signal_done(buf);

        // Overlap: Opart stores from previous m_block
        if (m_block > m_start_block) {
            auto& sO_prev = (buf == 0) ? sO1 : sO0;
            const int m_prev_start = m_start - kBlockM;
            #pragma unroll
            for (int v = 0; v < kVecsPerThreadO; ++v) {
                int idx = v * kNThreadsMMA + ctid;
                int i = idx / kVecsPerRowO;
                int c = (idx % kVecsPerRowO) * kElemsPerVecO;
                int row = m_prev_start + i;
                size_t off = ((((size_t)batch * S + row) * NH + head) * num_n_blocks_o
                           + n_block) * D + c;
                uint4 val = *reinterpret_cast<uint4 const*>(&sO_prev(i, c));
                *reinterpret_cast<uint4*>(Op_ptr + off) = val;
            }
        }

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

        SoftmaxT softmax(softmax_scale_log2);
        (void)softmax.template max_get_scale<true, true>(acc_s);
        softmax.template online_softmax<true, true>(acc_s);

        Tensor tOrP_acc = make_tensor(acc_s.data(),
            flash::convert_layout_acc_Aregs<TiledMmaPV>(acc_s.layout()));
        Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
        flash::convert_type_out(tOrP_acc, tOrP);

        Tensor acc_o = partition_fragment_C(tiled_mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});
        flash::gemm<true, -1>(tiled_mma_pv, tOrP, tOrV, acc_o);

        auto scores_scale = softmax.finalize();

        // LSE store
        {
            Tensor cO_lse = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
            Tensor tOcO_lse = thr_mma_pv.partition_C(cO_lse);
            int lane = ctid % 32;
            if (lane % 4 == 0) {
                Tensor tOcO_rc = make_tensor(tOcO_lse.data(),
                    flash::convert_layout_acc_rowcol(tOcO_lse.layout()));
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    int row_rel = get<0>(tOcO_rc(mi, _0{}));
                    int row = m_start + row_rel;
                    float lse = softmax.row_sum(mi);
                    size_t off_lse = (((size_t)batch * S + row) * NH + head)
                                   * params.num_n_blocks + n_block;
                    LSE_ptr[off_lse] = lse;
                }
            }
        }

        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(acc_o);
        softmax.rescale_o(acc_o, scores_scale);

        // STSM O to smem
        {
            Tensor rO = make_tensor_like<Element>(acc_o);
            flash::convert_type_out(acc_o, rO);
            Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
            auto& taccOsO_cur = (buf == 0) ? taccOsO0 : taccOsO1;
            cute::copy(smem_tiled_copy_O, taccOrO, taccOsO_cur);
            bar_consumer();
        }
    }

    // Post-loop: stores for the last m_block
    if (m_start_block < params.num_m_blocks) {
        const int last_m_start = (params.num_m_blocks - 1) * kBlockM;
        const int last_buf = (params.num_m_blocks - 1) & 1;
        auto& sO_last = (last_buf == 0) ? sO0 : sO1;
        #pragma unroll
        for (int v = 0; v < kVecsPerThreadO; ++v) {
            int idx = v * kNThreadsMMA + ctid;
            int i = idx / kVecsPerRowO;
            int c = (idx % kVecsPerRowO) * kElemsPerVecO;
            int row = last_m_start + i;
            size_t off = ((((size_t)batch * S + row) * NH + head) * num_n_blocks_o
                       + n_block) * D + c;
            uint4 val = *reinterpret_cast<uint4 const*>(&sO_last(i, c));
            *reinterpret_cast<uint4*>(Op_ptr + off) = val;
        }
    }
}

} // namespace proj_fused
