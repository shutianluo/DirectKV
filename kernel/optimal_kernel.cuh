/******************************************************************************
 * Optimal Fused Forward Kernel (SM90, bf16)
 *
 * FA1-style HBM running state + FA3 pipeline machinery.
 * Single-launch: outputs final O directly (no separate combine).
 *
 * Grid: (num_n_blocks, NH, B)  — KV-outer (preserves projection fusion).
 *
 * HBM running state (per (b, s_global, h)):
 *   O_run [B,S,NH,D] fp32 — un-normalized output, init 0
 *   m_run [B,S,NH]   fp32 — running max,           init -INF
 *   l_run [B,S,NH]   fp32 — running denominator,   init 0
 *   row_lock [B,S,NH] int — per-row spinlock,       init 0
 *   done_ct  [B,NH]   int — completed-CTA counter, init 0
 *
 * Per CTA:
 *   Phase 1 (project K/V) — same FA3 PipelineTmaAsync<2> as fused_kernel.cuh
 *   Phase 2 staggered Q traversal:
 *     - QK GEMM, softmax, PV GEMM (un-normalized)
 *     - Per-row online-softmax merge into HBM running state with row spinlock
 *   Last CTA per (b,h) detected via atomicAdd(done_ct):
 *     - Pointwise normalize O_run / l_run → final O (bf16)
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
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/reg_reconfig.h>

#include "proj_fused_kernel_traits_sm90.h"
#include "softmax.h"
#include "utils.h"

namespace optimal {

using namespace cute;

// 3D TMA helper (same PTX as fused_v3::tma_load_3d)
__device__ __forceinline__ void tma_load_3d(
    uint32_t smem_addr, CUtensorMap const* desc,
    int c0, int c1, int c2, uint32_t mbar_addr)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];\n"
        :: "r"(smem_addr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_addr));
}

enum class OptBarrier : uint32_t { OSmemReady = 0, MergeStatsReady = 1 };

using PipelineProj = cutlass::PipelineTmaAsync<2>;
using PipelineQ    = cutlass::PipelineTmaAsync<2>;

// ---------------------------------------------------------------------------
// Param struct
// ---------------------------------------------------------------------------
struct OptimalParams {
    void const* __restrict__ ptr_X;       // [B, S, H]                 bf16
    void const* __restrict__ ptr_Wk;      // [NH, D, H]                bf16
    void const* __restrict__ ptr_Wv;      // [NH, D, H]                bf16
    void const* __restrict__ ptr_Q;       // [B, S, NH, D]             bf16
    void*       __restrict__ ptr_O;       // [B, S, NH, D] FINAL OUT   bf16
    void*       __restrict__ ptr_Kc;      // [B, S, NH, D]             bf16
    void*       __restrict__ ptr_Vc;      // [B, S, NH, D]             bf16
    void const* __restrict__ ptr_Kpast;   // [B, S_past, NH, D]        bf16
    void const* __restrict__ ptr_Vpast;   // [B, S_past, NH, D]        bf16

    float* __restrict__ ptr_O_run;        // [B, S, NH, D]                fp32 (init 0)
    float* __restrict__ ptr_LSE_run;      // [B, S, NH]                   fp32 (init -INF)
    int*   __restrict__ ptr_mblock_lock;  // [B, num_m_blocks, NH]        int  (init 0); per-tile mutex
    int*   __restrict__ ptr_done_ct;      // [B, NH]                      int  (init 0)

    CUtensorMap const* tma_desc_Q;
    CUtensorMap const* tma_desc_X;
    CUtensorMap const* tma_desc_Wk;
    CUtensorMap const* tma_desc_Wv;

    int batch, seqlen, num_heads, hidden_dim, head_dim;
    int seqlen_past;
    int num_n_blocks_past;
    int num_n_blocks;
    int num_m_blocks;
    float softmax_scale;
    int is_causal;
};

// ---------------------------------------------------------------------------
// Init kernel — fills m_run = -INF (zero-able buffers handled via cudaMemsetAsync)
// ---------------------------------------------------------------------------
__global__ void fill_neg_inf_kernel(float* p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) p[idx] = -INFINITY;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
template <typename Traits>
__global__ __launch_bounds__(256, 1)
void optimal_fwd_kernel(OptimalParams params)
{
    using Element = typename Traits::Element;          // bfloat16_t
    constexpr int kBlockM        = Traits::kBlockM;
    constexpr int kBlockN        = Traits::kBlockN;
    constexpr int kHeadDim       = Traits::kHeadDim;
    constexpr int kHiddenChunk   = Traits::kHiddenChunk;
    constexpr int kNumProjChunks = Traits::kNumProjChunks;
    constexpr int kNThreadsMMA   = Traits::kNThreadsMMA;  // 128
    using SmemLayoutQ  = typename Traits::SmemLayoutQ;
    using SmemLayoutK  = typename Traits::SmemLayoutK;
    using SmemLayoutV  = typename Traits::SmemLayoutV;
    using SmemLayoutX  = typename Traits::SmemLayoutX;
    using SmemLayoutW  = typename Traits::SmemLayoutW;
    using SmemLayoutO  = typename Traits::SmemLayoutO;
    using TiledMmaProj = typename Traits::TiledMmaProj;
    using TiledMmaQK   = typename Traits::TiledMmaQK;
    using TiledMmaPV   = typename Traits::TiledMmaPV;

    const int n_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int tid     = threadIdx.x;
    const int wg_idx  = tid / 128;
    const int ctid    = tid - 128;

    if (wg_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
    } else {
        cutlass::arch::warpgroup_reg_alloc<240>();
    }

    const int S  = params.seqlen;
    const int NH = params.num_heads;
    const int S_past = params.seqlen_past;
    const int num_n_blocks_past = params.num_n_blocks_past;
    const int n_start = n_block * kBlockN;
    const bool is_past_kv = (n_block < num_n_blocks_past);
    const int n_start_new = n_start - S_past;

    auto* Kc_ptr  = reinterpret_cast<Element*>(params.ptr_Kc);
    auto* Vc_ptr  = reinterpret_cast<Element*>(params.ptr_Vc);
    auto* O_final_ptr = reinterpret_cast<Element*>(params.ptr_O);
    auto const* Kpast_ptr = reinterpret_cast<Element const*>(params.ptr_Kpast);
    auto const* Vpast_ptr = reinterpret_cast<Element const*>(params.ptr_Vpast);

    float* O_run_ptr      = params.ptr_O_run;
    float* LSE_run_ptr    = params.ptr_LSE_run;
    int*   mblock_lock_ptr = params.ptr_mblock_lock;
    int*   done_ct_ptr    = params.ptr_done_ct;

    // ------------------------------------------------------------------
    // Shared memory layout (same as fused_kernel.cuh)
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
    base = align128(base + cute::cosize(SmemLayoutO{}) * sizeof(Element));
    Element* sO1_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutO{}) * sizeof(Element));

    Tensor sQ0 = make_tensor(make_smem_ptr(sQ0_ptr), SmemLayoutQ{});
    Tensor sQ1 = make_tensor(make_smem_ptr(sQ1_ptr), SmemLayoutQ{});
    Tensor sK  = make_tensor(make_smem_ptr(sK_ptr),  SmemLayoutK{});
    Tensor sV  = make_tensor(make_smem_ptr(sV_ptr),  SmemLayoutV{});
    Tensor sO0 = make_tensor(make_smem_ptr(sO0_ptr), SmemLayoutO{});
    Tensor sO1 = make_tensor(make_smem_ptr(sO1_ptr), SmemLayoutO{});

    // Barrier region
    uintptr_t bar_base = align128(reinterpret_cast<uintptr_t>(sO1_ptr)
                                  + cute::cosize(SmemLayoutO{}) * sizeof(Element));
    bar_base = (bar_base + 15) & ~uintptr_t(15);
    auto* pipe_proj_smem = reinterpret_cast<PipelineProj::SharedStorage*>(bar_base);
    bar_base += sizeof(PipelineProj::SharedStorage);
    bar_base  = (bar_base + 15) & ~uintptr_t(15);
    auto* pipe_q_smem    = reinterpret_cast<PipelineQ::SharedStorage*>(bar_base);
    bar_base += sizeof(PipelineQ::SharedStorage);
    bar_base  = (bar_base + 15) & ~uintptr_t(15);

    // Per-row LSE scratch (kBlockM floats)
    float* sLSE_local = reinterpret_cast<float*>(bar_base);
    bar_base += kBlockM * sizeof(float);
    bar_base  = (bar_base + 15) & ~uintptr_t(15);

    constexpr int kTmaXBytes    = kBlockN  * kHiddenChunk * (int)sizeof(Element);
    constexpr int kTmaWBytes    = kHeadDim * kHiddenChunk * (int)sizeof(Element);
    constexpr int kTmaProjTotal = kTmaXBytes + 2 * kTmaWBytes;
    constexpr int kTmaHalfBytes = kBlockM * 64 * (int)sizeof(Element);

    // Pipeline construction (all 256 threads)
    PipelineProj::Params proj_p;
    proj_p.role             = (wg_idx == 0) ? PipelineProj::ThreadCategory::Producer
                                            : PipelineProj::ThreadCategory::Consumer;
    proj_p.transaction_bytes = kTmaProjTotal;
    proj_p.is_leader        = (wg_idx == 0 && tid == 0) ? 1u : 0u;
    proj_p.num_consumers    = static_cast<uint32_t>(kNThreadsMMA);
    PipelineProj pipeline_proj(*pipe_proj_smem, proj_p, Shape<_1,_1,_1>{});

    PipelineQ::Params q_p;
    q_p.role             = (wg_idx == 0) ? PipelineQ::ThreadCategory::Producer
                                         : PipelineQ::ThreadCategory::Consumer;
    q_p.transaction_bytes = 2u * static_cast<uint32_t>(kTmaHalfBytes);
    q_p.is_leader        = (wg_idx == 0 && tid == 0) ? 1u : 0u;
    q_p.num_consumers    = static_cast<uint32_t>(kNThreadsMMA);
    PipelineQ pipeline_q(*pipe_q_smem, q_p, Shape<_1,_1,_1>{});

    __syncthreads();

    // ==================================================================
    // PHASE 1: Project new KV or load past KV
    // ==================================================================

    if (is_past_kv) {
        constexpr int kElemsPerVec = 8;
        constexpr int kVecsPerRow  = kHeadDim / kElemsPerVec;
        constexpr int kTotalVecs   = kBlockN * kVecsPerRow;
        constexpr int kVecsPerThr  = (kTotalVecs + 255) / 256;

        size_t base_off = ((size_t)batch * S_past + n_start) * NH * kHeadDim
                        + (size_t)head * kHeadDim;
        auto const* K_tile = Kpast_ptr + base_off;
        auto const* V_tile = Vpast_ptr + base_off;

        #pragma unroll
        for (int v = 0; v < kVecsPerThr; ++v) {
            int idx = v * 256 + tid;
            if (idx < kTotalVecs) {
                int i = idx / kVecsPerRow;
                int d = (idx % kVecsPerRow) * kElemsPerVec;
                size_t row_off = (size_t)i * NH * kHeadDim + d;
                uint4 vk = *reinterpret_cast<uint4 const*>(K_tile + row_off);
                uint4 vv = *reinterpret_cast<uint4 const*>(V_tile + row_off);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    sK(i, d + k) = reinterpret_cast<Element const*>(&vk)[k];
                    sV(d + k, i) = reinterpret_cast<Element const*>(&vv)[k];
                }
            }
        }
    } else {
        Element* sX_p[2]  = { sQ0_ptr, sQ1_ptr };
        Element* sWk_p[2] = { sO0_ptr, sK_ptr  };
        Element* sWv_p[2] = { sO1_ptr, sV_ptr  };

        if (wg_idx == 0) {
            auto smem_pipe_write = cutlass::make_producer_start_state<PipelineProj>();
            for (int cs = 0; cs < kNumProjChunks; ++cs) {
                pipeline_proj.producer_acquire(smem_pipe_write);
                if (tid == 0) {
                    int h_off = cs * kHiddenChunk;
                    int stg   = smem_pipe_write.index();
                    uint32_t bar_addr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(pipeline_proj.producer_get_barrier(smem_pipe_write)));
                    tma_load_3d(
                        static_cast<uint32_t>(__cvta_generic_to_shared(sX_p[stg])),
                        params.tma_desc_X,  h_off, n_start_new, batch, bar_addr);
                    tma_load_3d(
                        static_cast<uint32_t>(__cvta_generic_to_shared(sWk_p[stg])),
                        params.tma_desc_Wk, h_off, 0,           head,  bar_addr);
                    tma_load_3d(
                        static_cast<uint32_t>(__cvta_generic_to_shared(sWv_p[stg])),
                        params.tma_desc_Wv, h_off, 0,           head,  bar_addr);
                }
                ++smem_pipe_write;
            }
        } else {
            TiledMmaProj tiled_mma_proj;
            auto thr_mma = tiled_mma_proj.get_thread_slice(ctid);
            Tensor acc_k = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
            Tensor acc_v = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
            clear(acc_k); clear(acc_v);

            auto smem_pipe_read = cutlass::PipelineState<2>{};
            for (int cs = 0; cs < kNumProjChunks; ++cs) {
                pipeline_proj.consumer_wait(smem_pipe_read);
                int stg = smem_pipe_read.index();
                Tensor sX_cur  = make_tensor(make_smem_ptr(sX_p[stg]),  SmemLayoutX{});
                Tensor sWk_cur = make_tensor(make_smem_ptr(sWk_p[stg]), SmemLayoutW{});
                Tensor sWv_cur = make_tensor(make_smem_ptr(sWv_p[stg]), SmemLayoutW{});
                Tensor tCsX  = thr_mma.partition_fragment_A(sX_cur);
                Tensor tCsWk = thr_mma.partition_fragment_B(sWk_cur);
                Tensor tCsWv = thr_mma.partition_fragment_B(sWv_cur);
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
                pipeline_proj.consumer_release(smem_pipe_read);
                ++smem_pipe_read;
            }

            Tensor tCsK = thr_mma.partition_C(sK);
            Tensor cV   = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
            Tensor tCcV = thr_mma.partition_C(cV);
            #pragma unroll
            for (int i = 0; i < size(acc_k); ++i) {
                tCsK(i) = Element(acc_k(i));
                sV(get<1>(tCcV(i)), get<0>(tCcV(i))) = Element(acc_v(i));
            }
        }
        __syncthreads();

        // Store sK → Kc, sV → Vc
        {
            constexpr int kEPV = 8;
            constexpr int kVPR = kHeadDim / kEPV;
            constexpr int kTV  = kBlockN * kVPR;
            constexpr int kVPT = (kTV + 255) / 256;
            #pragma unroll
            for (int v = 0; v < kVPT; ++v) {
                int idx = v * 256 + tid;
                if (idx < kTV) {
                    int i = idx / kVPR;
                    int d = (idx % kVPR) * kEPV;
                    int row = n_start_new + i;
                    if (row < S) {
                        size_t off = ((size_t)batch * S + row) * NH * kHeadDim
                                   + (size_t)head * kHeadDim + d;
                        uint4 kv_k, kv_v;
                        #pragma unroll
                        for (int k = 0; k < kEPV; ++k) {
                            reinterpret_cast<Element*>(&kv_k)[k] = sK(i, d + k);
                            reinterpret_cast<Element*>(&kv_v)[k] = sV(d + k, i);
                        }
                        *reinterpret_cast<uint4*>(Kc_ptr + off) = kv_k;
                        *reinterpret_cast<uint4*>(Vc_ptr + off) = kv_v;
                    }
                }
            }
        }
    }

    __syncthreads();

    const int new_n_idx     = n_block - num_n_blocks_past;
    const int m_start_block = (params.is_causal && !is_past_kv && new_n_idx > 0)
                              ? new_n_idx : 0;
    const int num_m_blocks  = params.num_m_blocks;

    // ==================================================================
    // PHASE 2: WG0 Q TMA producer / WG1 attention consumer + HBM merge
    // ==================================================================

    if (wg_idx == 0) {
        // ---- WG0: Q TMA producer with staggered iteration ----
        CUtensorMap const* tma_desc = params.tma_desc_Q;
        auto smem_pipe_write_q = cutlass::make_producer_start_state<PipelineQ>();
        int active_count = num_m_blocks - m_start_block;

        for (int k = 0; k < active_count; ++k) {
            int m_block = m_start_block + ((n_block + k) % active_count);
            pipeline_q.producer_acquire(smem_pipe_write_q);
            if (tid == 0) {
                int m_start = m_block * kBlockM;
                int idx = smem_pipe_write_q.index();
                Element* sq = (idx == 0) ? sQ0_ptr : sQ1_ptr;
                uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(sq));
                uint32_t s1 = s0 + kTmaHalfBytes;
                uint32_t mb = static_cast<uint32_t>(
                    __cvta_generic_to_shared(pipeline_q.producer_get_barrier(smem_pipe_write_q)));
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
            }
            ++smem_pipe_write_q;
        }
        // Producer is done; fall through to last-CTA finalize check below.
    } else {
        // ---- WG1: attention consumer + HBM merge ----
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

        int active_count = num_m_blocks - m_start_block;
        auto smem_pipe_read_q = cutlass::PipelineState<2>{};

        for (int k = 0; k < active_count; ++k) {
            int m_block = m_start_block + ((n_block + k) % active_count);
            int m_start = m_block * kBlockM;
            int o_buf   = smem_pipe_read_q.index();

            pipeline_q.consumer_wait(smem_pipe_read_q);

            // QK GEMM
            Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
            if (o_buf == 0) {
                flash::gemm<true, -1>(tiled_mma_qk, tSrQ0, tSrK, acc_s);
            } else {
                flash::gemm<true, -1>(tiled_mma_qk, tSrQ1, tSrK, acc_s);
            }
            cute::warpgroup_wait<0>();
            cute::warpgroup_fence_operand(acc_s);

            pipeline_q.consumer_release(smem_pipe_read_q);
            ++smem_pipe_read_q;

            // Causal mask (lower-right diagonal across [past + new])
            if (params.is_causal) {
                auto thread0_mma = TiledMmaQK{}.get_thread_slice(_0{});
                Tensor cS    = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
                Tensor tScS  = thr_mma_qk.partition_C(cS);
                Tensor t0ScS = thread0_mma.partition_C(cS);
                Tensor acc_s_rc  = make_tensor(acc_s.data(),
                    flash::convert_layout_acc_rowcol(acc_s.layout()));
                Tensor tScS_rc   = make_tensor(tScS.data(),
                    flash::convert_layout_acc_rowcol(tScS.layout()));
                Tensor t0ScS_rc  = make_tensor(t0ScS.data(),
                    flash::convert_layout_acc_rowcol(t0ScS.layout()));
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

            // softmax
            SoftmaxT softmax(softmax_scale_log2);
            (void)softmax.template max_get_scale<true, true>(acc_s);
            softmax.template online_softmax<true, true>(acc_s);

            // PV GEMM (un-normalized output for this CTA)
            Tensor tOrP_acc = make_tensor(acc_s.data(),
                flash::convert_layout_acc_Aregs<TiledMmaPV>(acc_s.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            flash::convert_type_out(tOrP_acc, tOrP);

            Tensor acc_o = partition_fragment_C(tiled_mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});
            flash::gemm<true, -1>(tiled_mma_pv, tOrP, tOrV, acc_o);

            // Finalize softmax: warp-reduces row_sum and turns row_sum into LSE.
            // scores_scale = 1/l_local (un-normalized denominator inverse).
            auto scores_scale = softmax.finalize();

            cute::warpgroup_wait<0>();
            cute::warpgroup_fence_operand(acc_o);
            // Normalize acc_o (multiply by 1/l_local) — match existing fused kernel pattern.
            softmax.rescale_o(acc_o, scores_scale);

            // (a) Persist LSE_local per row to smem (row-leader pattern).
            //     After finalize(), softmax.row_sum(mi) holds LSE_local in natural log:
            //       LSE_local = softmax_scale * m_local + log(l_local)
            {
                Tensor cO   = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
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

            // (b) STSM acc_o → sO[o_buf]
            {
                Tensor rO = make_tensor_like<Element>(acc_o);
                flash::convert_type_out(acc_o, rO);
                Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
                auto& taccOsO_cur = (o_buf == 0) ? taccOsO0 : taccOsO1;
                cute::copy(smem_tiled_copy_O, taccOrO, taccOsO_cur);
            }

            // Sync WG1: stats and sO ready for the merge phase.
            cutlass::arch::NamedBarrier::sync(
                static_cast<uint32_t>(kNThreadsMMA),
                static_cast<uint32_t>(OptBarrier::OSmemReady));

            // (c) Per-tile online-softmax merge into HBM running state.
            //     ONE lock per (b, m_block, h). ctid==0 acquires; all 128 threads merge
            //     cooperatively while the lock is held; ctid==0 releases.
            //     Threading: 2 threads/row × 64 rows = 128. d_half ∈ {0,1}.
            {
                int mblock_lock_id = (batch * num_m_blocks + m_block) * NH + head;

                // ctid==0 acquires the per-tile lock (single-thread spin → no multi-lock deadlock)
                if (ctid == 0) {
                    while (atomicCAS(&mblock_lock_ptr[mblock_lock_id], 0, 1) != 0) { /* spin */ }
                    __threadfence();
                }
                cutlass::arch::NamedBarrier::sync(
                    static_cast<uint32_t>(kNThreadsMMA),
                    static_cast<uint32_t>(OptBarrier::MergeStatsReady));

                // All 128 threads merge in parallel.
                int row_in_tile = ctid >> 1;       // 0..63
                int d_half      = ctid & 1;
                int s_global    = m_start + row_in_tile;

                if (s_global < S) {
                    int row_id = ((batch * S + s_global) * NH + head);

                    float LSE_local_val = sLSE_local[row_in_tile];
                    float LSE_old = LSE_run_ptr[row_id];

                    // LSE-based merge
                    float alpha, beta, LSE_new;
                    float LSE_max = fmaxf(LSE_old, LSE_local_val);
                    if (LSE_max == -INFINITY) {
                        alpha = 0.f; beta = 0.f; LSE_new = -INFINITY;
                    } else {
                        float a_old = (LSE_old       == -INFINITY) ? 0.f : __expf(LSE_old       - LSE_max);
                        float a_loc = (LSE_local_val == -INFINITY) ? 0.f : __expf(LSE_local_val - LSE_max);
                        float Z = a_old + a_loc;
                        LSE_new = LSE_max + __logf(Z);
                        alpha = (LSE_old       == -INFINITY) ? 0.f : __expf(LSE_old       - LSE_new);
                        beta  = (LSE_local_val == -INFINITY) ? 0.f : __expf(LSE_local_val - LSE_new);
                    }

                    // Each thread updates D/2 = 64 elements
                    constexpr int kHalfD = kHeadDim / 2;
                    int d_start = d_half * kHalfD;
                    auto& sO_cur_t = (o_buf == 0) ? sO0 : sO1;

                    #pragma unroll
                    for (int d = 0; d < kHalfD; d += 4) {
                        int d_global = d_start + d;
                        float4 o_old = *reinterpret_cast<const float4*>(
                            &O_run_ptr[row_id * kHeadDim + d_global]);
                        Element so0_ = sO_cur_t(row_in_tile, d_global + 0);
                        Element so1_ = sO_cur_t(row_in_tile, d_global + 1);
                        Element so2_ = sO_cur_t(row_in_tile, d_global + 2);
                        Element so3_ = sO_cur_t(row_in_tile, d_global + 3);
                        float4 o_new;
                        o_new.x = alpha * o_old.x + beta * (float)so0_;
                        o_new.y = alpha * o_old.y + beta * (float)so1_;
                        o_new.z = alpha * o_old.z + beta * (float)so2_;
                        o_new.w = alpha * o_old.w + beta * (float)so3_;
                        *reinterpret_cast<float4*>(&O_run_ptr[row_id * kHeadDim + d_global]) = o_new;
                    }

                    // Only one thread per row writes the new LSE
                    if (d_half == 0) {
                        LSE_run_ptr[row_id] = LSE_new;
                    }
                }

                // Wait for all merges to complete, then ctid==0 releases the lock.
                cutlass::arch::NamedBarrier::sync(
                    static_cast<uint32_t>(kNThreadsMMA),
                    static_cast<uint32_t>(OptBarrier::OSmemReady));
                if (ctid == 0) {
                    __threadfence();
                    atomicExch(&mblock_lock_ptr[mblock_lock_id], 0);
                }
                // Sync WG1 before next iteration's STSM (uses sO[o_buf] ping-pong)
                cutlass::arch::NamedBarrier::sync(
                    static_cast<uint32_t>(kNThreadsMMA),
                    static_cast<uint32_t>(OptBarrier::MergeStatsReady));
            }
        }
    }

    // ==================================================================
    // Last-CTA finalize: pointwise normalize O_run / l_run → final O (bf16)
    // ==================================================================
    __threadfence();
    __syncthreads();

    __shared__ int s_is_last;
    if (tid == 0) {
        int prev = atomicAdd(&done_ct_ptr[batch * NH + head], 1);
        s_is_last = (prev == params.num_n_blocks - 1) ? 1 : 0;
    }
    __syncthreads();

    if (s_is_last) {
        // O_run is already the final NORMALIZED output (LSE-based incremental merge).
        // Just cast to bf16 and write.
        int total = S * kHeadDim;
        for (int idx = tid; idx < total; idx += 256) {
            int s = idx / kHeadDim;
            int d = idx % kHeadDim;
            int row_id = (batch * S + s) * NH + head;
            float o = O_run_ptr[row_id * kHeadDim + d];
            size_t off = (((size_t)batch * S + s) * NH + head) * kHeadDim + d;
            O_final_ptr[off] = (Element)o;
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
template <typename Traits>
cudaError_t launch_optimal(OptimalParams p, cudaStream_t stream = 0)
{
    using Element = typename Traits::Element;

    // (1) Initialize HBM scratch
    int n_row = p.batch * p.seqlen * p.num_heads;
    size_t n_O = (size_t)n_row * Traits::kHeadDim;
    int n_bh  = p.batch * p.num_heads;
    int n_lock = p.batch * p.num_m_blocks * p.num_heads;

    cudaError_t err;
    err = cudaMemsetAsync(p.ptr_O_run,       0, n_O   * sizeof(float), stream); if (err) return err;
    err = cudaMemsetAsync(p.ptr_mblock_lock, 0, n_lock* sizeof(int),   stream); if (err) return err;
    err = cudaMemsetAsync(p.ptr_done_ct,     0, n_bh  * sizeof(int),   stream); if (err) return err;
    {
        int tpb = 256, blocks = (n_row + tpb - 1) / tpb;
        fill_neg_inf_kernel<<<blocks, tpb, 0, stream>>>(p.ptr_LSE_run, n_row);
    }

    // (2) Smem size — same layout as fused kernel + sm_local/sl_local (kBlockM*4 each)
    auto r128 = [](size_t s) { return (s + 127) & ~size_t(127); };
    size_t smem = 128;
    smem += r128(cute::cosize(typename Traits::SmemLayoutQ{}) * sizeof(Element)) * 2;
    smem += r128(cute::cosize(typename Traits::SmemLayoutK{}) * sizeof(Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutV{}) * sizeof(Element));
    smem += r128(cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element)) * 2;
    smem += 256;  // pipeline barrier region
    smem += 2 * Traits::kBlockM * sizeof(float) + 32;  // sm_local + sl_local

    dim3 grid(p.num_n_blocks, p.num_heads, p.batch);
    dim3 block(256);
    auto* kernel = &optimal_fwd_kernel<Traits>;
    err = cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (err != cudaSuccess) return err;
    kernel<<<grid, block, smem, stream>>>(p);
    return cudaGetLastError();
}

} // namespace optimal
