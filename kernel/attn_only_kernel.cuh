/******************************************************************************
 * Phase 2 — Attention-Only Kernel (SM90)
 *
 * Loads precomputed K/V from global memory into smem, then runs the
 * attention loop (Q blocks) for one KV tile per CTA.
 *
 * Thread organization (256 threads / 2 warpgroups):
 *   WG0 (tid 0-127):   TMA Q producer.  Loads Q tiles via TMA bulk load.
 *   WG1 (tid 128-255): Attention consumer.  QK WGMMA → softmax → PV WGMMA
 *                      → STSM → uint4 Opart store.
 *
 * K/V startup: all 256 threads issue synchronous ld.global.v4 (uint4 dereference)
 * with inline V-transpose into MN-major sV.  No TMA, no cp.async.
 *
 * Opart output: raw uint4 stores (no TMA — Phase 4 optimization).
 *
 * Correctness gates:
 *   O max_abs_err < 1e-2  vs. F.scaled_dot_product_attention(Q, K, V)
 *   LSE max_abs_err < 1e-3 vs. reference
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

namespace attn_only {

using namespace cute;

// ---------------------------------------------------------------------------
// Param struct
// ---------------------------------------------------------------------------
struct AttnOnlyParams {
    void const* __restrict__ ptr_K;     // [B, S_full, NH, D]  — full K (past + new)
    void const* __restrict__ ptr_V;     // [B, S_full, NH, D]  — full V (past + new)
    void const* __restrict__ ptr_Q;     // [B, S_new,  NH, D]
    void*       __restrict__ ptr_Opart; // [B, S_new, NH, num_n_blocks, D]
    float*      __restrict__ ptr_LSE;   // [B, S_new, NH, num_n_blocks]
    CUtensorMap const* tma_desc_Q;

    int batch, seqlen_q, seqlen_kv;    // seqlen_q = S_new, seqlen_kv = S_past + S_new
    int num_heads, head_dim;
    int num_n_blocks;                   // ceil(seqlen_kv / kBlockN)
    int num_m_blocks;                   // ceil(seqlen_q / kBlockM)
    int seqlen_past;                    // 0 if no past KV
    float softmax_scale;
    int is_causal;
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <typename Traits>
__global__ __launch_bounds__(256, 1)
void attn_only_fwd_kernel(AttnOnlyParams params)
{
    using Element = typename Traits::Element;
    constexpr int kBlockM   = Traits::kBlockM;
    constexpr int kBlockN   = Traits::kBlockN;
    constexpr int kHeadDim  = Traits::kHeadDim;
    constexpr int kNThreadsMMA = Traits::kNThreadsMMA;  // 128
    using SmemLayoutQ  = typename Traits::SmemLayoutQ;
    using SmemLayoutK  = typename Traits::SmemLayoutK;
    using SmemLayoutV  = typename Traits::SmemLayoutV;
    using TiledMmaQK   = typename Traits::TiledMmaQK;
    using TiledMmaPV   = typename Traits::TiledMmaPV;

    const int n_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int tid     = threadIdx.x;

    const int wg_idx = tid / 128;
    const int ctid   = tid - 128;      // consumer-local [0,127], valid only for WG1

    const int S_q  = params.seqlen_q;
    const int S_kv = params.seqlen_kv;
    const int S_past = params.seqlen_past;
    const int NH   = params.num_heads;
    const int n_start = n_block * kBlockN;

    auto const* K_ptr  = reinterpret_cast<Element const*>(params.ptr_K);
    auto const* V_ptr  = reinterpret_cast<Element const*>(params.ptr_V);
    auto* Op_ptr  = reinterpret_cast<Element*>(params.ptr_Opart);
    float* LSE_ptr = params.ptr_LSE;

    // ------------------------------------------------------------------
    // Shared memory layout:
    //   sQ0 [kBlockM, kHeadDim]   Q ping   16 KB
    //   sQ1 [kBlockM, kHeadDim]   Q pong   16 KB
    //   sK  [kBlockN, kHeadDim]   K         16 KB
    //   sV  [kHeadDim, kBlockN]   V MN-maj  16 KB
    //   sO0 [kBlockM, kHeadDim]   O ping    16 KB
    //   sO1 [kBlockM, kHeadDim]   O pong    16 KB
    //   mbar_Q[2]                 TMA barriers
    //   smem_done[2]              consumer-done flags
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto align128 = [](uintptr_t p) -> uintptr_t { return (p + 127) & ~uintptr_t(127); };
    uintptr_t base = align128(reinterpret_cast<uintptr_t>(smem_raw));

    Element* sQ0_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));
    Element* sQ1_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));
    Element* sK_ptr  = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutK{}) * sizeof(Element));
    Element* sV_ptr  = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutV{}) * sizeof(Element));
    Element* sO0_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));
    Element* sO1_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));

    using SmemLayoutO = typename Traits::SmemLayoutO;
    Tensor sQ0 = make_tensor(make_smem_ptr(sQ0_ptr), SmemLayoutQ{});
    Tensor sQ1 = make_tensor(make_smem_ptr(sQ1_ptr), SmemLayoutQ{});
    Tensor sK  = make_tensor(make_smem_ptr(sK_ptr),  SmemLayoutK{});
    Tensor sV  = make_tensor(make_smem_ptr(sV_ptr),  SmemLayoutV{});
    Tensor sO0 = make_tensor(make_smem_ptr(sO0_ptr), SmemLayoutO{});
    Tensor sO1 = make_tensor(make_smem_ptr(sO1_ptr), SmemLayoutO{});

    // mbarrier and consumer-done flags placed after sO1
    uint64_t* mbar_Q = reinterpret_cast<uint64_t*>(
        align128(reinterpret_cast<uintptr_t>(sO1_ptr)
                 + cute::cosize(SmemLayoutO{}) * sizeof(Element)));
    // align to 8 bytes
    mbar_Q = reinterpret_cast<uint64_t*>((reinterpret_cast<uintptr_t>(mbar_Q) + 7) & ~uintptr_t(7));
    int* smem_done = reinterpret_cast<int*>(mbar_Q + 2);

    if (tid == 0) {
        uint32_t mb0 = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_Q[0]));
        uint32_t mb1 = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_Q[1]));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mb0), "r"(1));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mb1), "r"(1));
        smem_done[0] = 1;
        smem_done[1] = 1;
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 1: load K and V from global into sK / sV (all 256 threads)
    // Synchronous uint4 loads with inline V-transpose (no cp.async).
    // ------------------------------------------------------------------
    {
        constexpr int kElemsPerVec = 8;
        constexpr int kVecsPerRow  = kHeadDim / kElemsPerVec;
        constexpr int kTotalVecs   = kBlockN * kVecsPerRow;
        constexpr int kVecsPerThr  = (kTotalVecs + 255) / 256;

        // K/V global pointers for this KV tile
        // Layout: [B, S_kv, NH, D]  —  stride: D, NH*D, S_kv*NH*D
        size_t base_off = ((size_t)batch * S_kv + n_start) * NH * kHeadDim
                        + (size_t)head * kHeadDim;

        auto const* K_tile = K_ptr + base_off;
        auto const* V_tile = V_ptr + base_off;

        #pragma unroll
        for (int v = 0; v < kVecsPerThr; ++v) {
            int idx = v * 256 + tid;
            if (idx < kTotalVecs) {
                int i = idx / kVecsPerRow;              // row in KV tile
                int d = (idx % kVecsPerRow) * kElemsPerVec;  // col base
                // stride across tokens: NH*D per token
                size_t row_off = (size_t)i * NH * kHeadDim + d;
                uint4 vk = *reinterpret_cast<uint4 const*>(K_tile + row_off);
                uint4 vv = *reinterpret_cast<uint4 const*>(V_tile + row_off);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    sK(i, d + k) = reinterpret_cast<Element const*>(&vk)[k];
                    // V transpose: global V[i][d+k] → sV[d+k][i]
                    sV(d + k, i) = reinterpret_cast<Element const*>(&vv)[k];
                }
            }
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Causal: first m_block that overlaps this KV tile (for new-KV blocks)
    // For past-KV blocks (n_start < S_past) all Q blocks attend to them.
    // For new-KV blocks: only Q blocks where m_start <= n_start+kBlockN-1.
    // ------------------------------------------------------------------
    const bool is_past_n = (n_start < S_past);
    const int new_n_idx  = n_block - (S_past / kBlockN);
    const int m_start_block = (params.is_causal && !is_past_n && new_n_idx > 0)
                              ? new_n_idx : 0;

    // consumer barrier helper
    auto bar_consumer = [](){ asm volatile("bar.sync 1, 128;\n" ::); };

    // ------------------------------------------------------------------
    // PRODUCER (WG0): TMA Q loader
    // ------------------------------------------------------------------
    if (wg_idx == 0) {
        CUtensorMap const* tma_desc = params.tma_desc_Q;
        constexpr int kTmaHalfBytes = kBlockM * 64 * (int)sizeof(Element);

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
                smem_done[buf] = 0;
            }
        };

        for (int m_block = m_start_block; m_block < params.num_m_blocks; ++m_block) {
            int buf = m_block & 1;
            if (tid == 0) {
                while (atomicAdd(&smem_done[buf], 0) == 0) { __nanosleep(32); }
            }
            __syncwarp(0xffffffff);
            producer_load_Q(m_block * kBlockM, buf);
        }
        return;
    }

    // ------------------------------------------------------------------
    // CONSUMER (WG1): attention compute
    // ------------------------------------------------------------------
    float const softmax_scale_log2 = params.softmax_scale * float(M_LOG2E);
    constexpr int kNRows = 2 * (kBlockM / 64);
    using SoftmaxT = flash::Softmax<kNRows, 0>;

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

    constexpr int kElemsPerVecO = 8;
    constexpr int kVecsPerRowO  = kHeadDim / kElemsPerVecO;
    constexpr int kTotalVecsO   = kBlockM * kVecsPerRowO;
    constexpr int kVecsPerThrO  = kTotalVecsO / kNThreadsMMA;

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

    auto consumer_signal_done = [&](int buf) {
        bar_consumer();
        if (ctid == 0) { atomicExch(&smem_done[buf], 1); }
    };

    // Write LSE=-INF for causal-skipped m_blocks
    for (int m_block = 0; m_block < m_start_block; ++m_block) {
        int m_start = m_block * kBlockM;
        for (int i = ctid; i < kBlockM; i += kNThreadsMMA) {
            int row = m_start + i;
            size_t off_lse = (((size_t)batch * S_q + row) * NH + head)
                           * params.num_n_blocks + n_block;
            LSE_ptr[off_lse] = -INFINITY;
        }
    }

    // Main attention loop
    for (int m_block = m_start_block; m_block < params.num_m_blocks; ++m_block) {
        int m_start = m_block * kBlockM;
        int buf = m_block & 1;

        tma_wait_Q(buf);
        bar_consumer();

        // GEMM-I: S = Q @ K^T
        Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
        if (buf == 0) {
            flash::gemm<true, -1>(tiled_mma_qk, tSrQ0, tSrK, acc_s);
        } else {
            flash::gemm<true, -1>(tiled_mma_qk, tSrQ1, tSrK, acc_s);
        }
        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(acc_s);

        consumer_signal_done(buf);

        // Overlap: Opart stores from previous m_block
        if (m_block > m_start_block) {
            auto& sO_prev = (buf == 0) ? sO1 : sO0;
            int m_prev_start = m_start - kBlockM;
            #pragma unroll
            for (int v = 0; v < kVecsPerThrO; ++v) {
                int idx = v * kNThreadsMMA + ctid;
                int i = idx / kVecsPerRowO;
                int c = (idx % kVecsPerRowO) * kElemsPerVecO;
                int row = m_prev_start + i;
                size_t off = ((((size_t)batch * S_q + row) * NH + head)
                           * params.num_n_blocks + n_block) * kHeadDim + c;
                uint4 val = *reinterpret_cast<uint4 const*>(&sO_prev(i, c));
                *reinterpret_cast<uint4*>(Op_ptr + off) = val;
            }
        }

        // Causal mask
        if (params.is_causal) {
            auto thread0_mma = TiledMmaQK{}.get_thread_slice(_0{});
            Tensor cS     = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
            Tensor tScS   = thr_mma_qk.partition_C(cS);
            Tensor t0ScS  = thread0_mma.partition_C(cS);
            Tensor acc_s_rc  = make_tensor(acc_s.data(),
                flash::convert_layout_acc_rowcol(acc_s.layout()));
            Tensor tScS_rc   = make_tensor(tScS.data(),
                flash::convert_layout_acc_rowcol(tScS.layout()));
            Tensor t0ScS_rc  = make_tensor(t0ScS.data(),
                flash::convert_layout_acc_rowcol(t0ScS.layout()));
            int thread_col_offset = get<1>(tScS_rc(_0{}, _0{}));
            // causal: Q[m_start + row_rel] can attend to K up to S_past + m_start + row_rel
            // KV column index = n_start + col_rel
            // Allow if: n_start + col_rel <= S_past + m_start + row_rel
            // i.e. col_rel <= S_past + m_start + row_rel - n_start
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
            Tensor cO_lse   = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
            Tensor tOcO_lse = thr_mma_pv.partition_C(cO_lse);
            int lane = ctid % 32;
            if (lane % 4 == 0) {
                Tensor tOcO_rc = make_tensor(tOcO_lse.data(),
                    flash::convert_layout_acc_rowcol(tOcO_lse.layout()));
                #pragma unroll
                for (int mi = 0; mi < kNRows; ++mi) {
                    int row_rel = get<0>(tOcO_rc(mi, _0{}));
                    int row     = m_start + row_rel;
                    float lse   = softmax.row_sum(mi);
                    size_t off_lse = (((size_t)batch * S_q + row) * NH + head)
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

    // Post-loop: store last m_block's Opart
    if (m_start_block < params.num_m_blocks) {
        int last_m_start = (params.num_m_blocks - 1) * kBlockM;
        int last_buf     = (params.num_m_blocks - 1) & 1;
        auto& sO_last    = (last_buf == 0) ? sO0 : sO1;
        #pragma unroll
        for (int v = 0; v < kVecsPerThrO; ++v) {
            int idx = v * kNThreadsMMA + ctid;
            int i   = idx / kVecsPerRowO;
            int c   = (idx % kVecsPerRowO) * kElemsPerVecO;
            int row = last_m_start + i;
            size_t off = ((((size_t)batch * S_q + row) * NH + head)
                       * params.num_n_blocks + n_block) * kHeadDim + c;
            uint4 val = *reinterpret_cast<uint4 const*>(&sO_last(i, c));
            *reinterpret_cast<uint4*>(Op_ptr + off) = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------
template <typename Traits>
cudaError_t launch_attn_only(
    void const* ptr_K, void const* ptr_V, void const* ptr_Q,
    void* ptr_Opart, float* ptr_LSE,
    CUtensorMap const* tma_desc_Q,
    int batch, int seqlen_q, int seqlen_kv, int num_heads,
    int seqlen_past, float softmax_scale, bool is_causal,
    cudaStream_t stream = 0)
{
    AttnOnlyParams p;
    p.ptr_K = ptr_K; p.ptr_V = ptr_V; p.ptr_Q = ptr_Q;
    p.ptr_Opart = ptr_Opart; p.ptr_LSE = ptr_LSE;
    p.tma_desc_Q = tma_desc_Q;
    p.batch = batch; p.seqlen_q = seqlen_q; p.seqlen_kv = seqlen_kv;
    p.num_heads = num_heads; p.head_dim = Traits::kHeadDim;
    p.seqlen_past = seqlen_past;
    p.num_n_blocks = (seqlen_kv + Traits::kBlockN - 1) / Traits::kBlockN;
    p.num_m_blocks = (seqlen_q  + Traits::kBlockM - 1) / Traits::kBlockM;
    p.softmax_scale = softmax_scale;
    p.is_causal = is_causal ? 1 : 0;

    dim3 grid(p.num_n_blocks, num_heads, batch);
    dim3 block(256);

    auto r128 = [](size_t s) { return (s + 127) & ~size_t(127); };
    size_t smem = 128;
    smem += r128(cute::cosize(typename Traits::SmemLayoutQ{}) * sizeof(typename Traits::Element)) * 2;
    smem += r128(cute::cosize(typename Traits::SmemLayoutK{}) * sizeof(typename Traits::Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutV{}) * sizeof(typename Traits::Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(typename Traits::Element)) * 2;
    smem += 256;  // mbar_Q[2] + smem_done[2] + alignment

    auto* kernel = &attn_only_fwd_kernel<Traits>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (err != cudaSuccess) return err;
    kernel<<<grid, block, smem, stream>>>(p);
    return cudaGetLastError();
}

} // namespace attn_only
