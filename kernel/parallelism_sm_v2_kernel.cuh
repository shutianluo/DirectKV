/******************************************************************************
 * SM-Parallel v2 — TMA-Pipelined Fused Projection + Attention (SM90, bf16)
 *
 * Grid=(NH, B): one CTA per (head, batch). No spinlocks.
 * WG0 (tid 0..127): TMA producer for all data movement.
 * WG1 (tid 128..255): WGMMA consumer + HBM merge.
 *
 * Key improvements over SMP v1:
 *   1. TMA replaces uint4 scatter-loops → frees ALU, enables latency hiding.
 *   2. Double-buffered X pipeline (PipelineProj): TMA for X chunk cs+1 overlaps
 *      WGMMA on X chunk cs.
 *   3. Double-buffered Q pipeline (PipelineQ): TMA for Q[m+1] overlaps QK GEMM
 *      on Q[m].
 *   4. Wk/Wv loaded ONCE at startup into persistent smem (sWk_full/sWv_full),
 *      saving (num_n_blocks-1)*64KB HBM bandwidth vs OPT's per-n_block load.
 *   5. No done_ct / spinlock: this CTA exclusively owns all rows for (batch,head).
 *
 * PipelineProj.transaction_bytes = kTmaXBytes = 8KB   (X only; NOT OPT's 40KB)
 * PipelineQ.transaction_bytes    = 2*kTmaQHalfBytes = 16KB
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
#include <cutlass/gemm/collective/builders/sm90_common.inl>

#include "proj_fused_kernel_traits_sm90.h"
#include "softmax.h"
#include "utils.h"

namespace sm_parallel_v2 {

using namespace cute;

// ---------------------------------------------------------------------------
// TMA helpers (same PTX as optimal / proj_fused namespaces)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tma_load_3d(
    uint32_t smem_addr, CUtensorMap const* desc,
    int c0, int c1, int c2, uint32_t mbar_addr)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];\n"
        :: "r"(smem_addr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_addr));
}

__device__ __forceinline__ void tma_load_4d(
    uint32_t smem_addr, CUtensorMap const* desc,
    int c0, int c1, int c2, int c3, uint32_t mbar_addr)
{
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
        :: "r"(smem_addr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar_addr));
}

__device__ __forceinline__ void mbar_init(uint64_t* mbar, int count) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(count));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* mbar, int bytes) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, int phase) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    // .shared::cta scope provides proxy-acquire fence for cp.async.bulk writes (SM90 requirement)
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "MWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@!P bra MWAIT_%=;\n"
        "}\n" :: "r"(a), "r"(phase));
}

// ---------------------------------------------------------------------------
// Pipeline types
// ---------------------------------------------------------------------------
using PipelineProj = cutlass::PipelineTmaAsync<2>;
using PipelineQ    = cutlass::PipelineTmaAsync<2>;

// ---------------------------------------------------------------------------
// Barrier enum for WG1-internal synchronization
// ---------------------------------------------------------------------------
enum class SmpV2Barrier : uint32_t { OSmemReady = 0, MergeStatsReady = 1 };

// ---------------------------------------------------------------------------
// Params struct
// ---------------------------------------------------------------------------
struct SmParallelV2Params {
    void const* ptr_X;        // [B, S_new, kHiddenDim] bf16
    void const* ptr_Wk;       // [NH, kHeadDim, kHiddenDim] bf16
    void const* ptr_Wv;       // [NH, kHeadDim, kHiddenDim] bf16
    void const* ptr_Q;        // [B, S_q, NH, kHeadDim] bf16
    void*       ptr_O;        // [B, S_q, NH, kHeadDim] bf16 output
    float*      ptr_O_run;    // [B, S_q, NH, kHeadDim] fp32, init 0
    float*      ptr_LSE_run;  // [B, S_q, NH] fp32, init -INF

    int batch, seqlen_new, seqlen_q, num_heads, head_dim, hidden_dim;
    int num_m_blocks, num_n_blocks;
    float softmax_scale;
    int is_causal;

    // TMA descriptors
    CUtensorMap const* tma_X;   // 3D: (kHiddenDim, S_new, B), box={kHC, kBN, 1}
    CUtensorMap const* tma_Wk;  // 3D: (kHiddenDim, kHeadDim, NH), box={kHC, kD, 1}
    CUtensorMap const* tma_Wv;  // 3D: (kHiddenDim, kHeadDim, NH), box={kHC, kD, 1}
    CUtensorMap const* tma_Q;   // 4D: (kHeadDim, NH, S_q, B), box={64,1,kBM,1}

    // Debug: set non-null to collect per-checkpoint arrival counts for wave-2 CTAs.
    // Layout: [0..6] = WG0 checkpoints (tid==0), [7..13] = WG1 checkpoints (tid==128).
    uint32_t*   debug_ctrs;
};

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
template <typename Traits>
__global__ __launch_bounds__(256, 1)
void sm_parallel_v2_fwd_kernel(SmParallelV2Params params)
{
    using Element = typename Traits::Element;
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

    // Full-size W layout: [kHeadDim, kHiddenDim] tiled from SmemLayoutW
    using SmemLayoutAtomW_full = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element,
            cute::Int<Traits::kHeadDim>, cute::Int<Traits::kHiddenDim>>());
    using SmemLayoutW_full = decltype(cute::tile_to_shape(
        SmemLayoutAtomW_full{},
        cute::make_shape(cute::Int<Traits::kHeadDim>{}, cute::Int<Traits::kHiddenDim>{})));

    const int head  = blockIdx.x;
    const int batch = blockIdx.y;
    const int tid   = threadIdx.x;
    const int wg_idx = tid / 128;
    const int ctid   = tid - 128;   // valid only when wg_idx == 1
    const int linearIdx = blockIdx.y * gridDim.x + blockIdx.x;
    const bool dbg_wg0 = (params.debug_ctrs != nullptr && linearIdx >= 132 && tid == 0);
    const bool dbg_wg1 = (params.debug_ctrs != nullptr && linearIdx >= 132 && tid == 128);
    if (dbg_wg0) atomicAdd(&params.debug_ctrs[6], 1u);  // CP6: kernel entry

    const int NH    = params.num_heads;
    const int S_q   = params.seqlen_q;
    const int S_new = params.seqlen_new;

    // setmaxnreg dec/inc removed: dec<40> was a NOP (C7507) because WG0 uses >40 regs,
    // and the paired inc was potentially leaving SM register-tracker state inconsistent
    // across CTA waves, causing wave-2 CTAs to hang.

    auto* O_final_ptr = reinterpret_cast<Element*>(params.ptr_O);
    float* O_run_ptr  = params.ptr_O_run;
    float* LSE_run_ptr = params.ptr_LSE_run;

    // ------------------------------------------------------------------
    // Shared memory layout:
    //   sQ0       [kBlockM, kHeadDim]  SmemLayoutQ   16KB  X staging / Q buf 0
    //   sQ1       [kBlockM, kHeadDim]  SmemLayoutQ   16KB  X staging / Q buf 1
    //   sWk_full  [kHeadDim, kHidDim]  SmemLayoutW_full 32KB persistent Wk
    //   sWv_full  [kHeadDim, kHidDim]  SmemLayoutW_full 32KB persistent Wv
    //   sK        [kBlockN, kHeadDim]  SmemLayoutK   16KB  projected K tile
    //   sV        [kHeadDim, kBlockN]  SmemLayoutV   16KB  projected V tile
    //   sO0       [kBlockM, kHeadDim]  SmemLayoutO   16KB  output staging buf 0
    //   sO1       [kBlockM, kHeadDim]  SmemLayoutO   16KB  output staging buf 1
    //   startup_mbar uint64_t           8B
    //   pipe_proj_smem PipelineProj::SharedStorage
    //   pipe_q_smem    PipelineQ::SharedStorage
    //   sLSE_local float[kBlockM]       256B
    //   Total: ~160KB
    // ------------------------------------------------------------------
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto align128 = [](uintptr_t p) -> uintptr_t { return (p + 127) & ~uintptr_t(127); };
    uintptr_t base = align128(reinterpret_cast<uintptr_t>(smem_raw));

    Element* sQ0_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));
    Element* sQ1_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutQ{}) * sizeof(Element));
    Element* sWk_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutW_full{}) * sizeof(Element));
    Element* sWv_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutW_full{}) * sizeof(Element));
    Element* sK_ptr  = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutK{}) * sizeof(Element));
    Element* sV_ptr  = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutV{}) * sizeof(Element));
    Element* sO0_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutO{}) * sizeof(Element));
    Element* sO1_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutO{}) * sizeof(Element));

    // Barriers
    base = (base + 7) & ~uintptr_t(7);   // 8-byte align for uint64_t
    uint64_t* startup_mbar = reinterpret_cast<uint64_t*>(base);
    base += 8;
    base = (base + 15) & ~uintptr_t(15);
    auto* pipe_proj_smem = reinterpret_cast<PipelineProj::SharedStorage*>(base);
    base += sizeof(PipelineProj::SharedStorage);
    base = (base + 15) & ~uintptr_t(15);
    auto* pipe_q_smem = reinterpret_cast<PipelineQ::SharedStorage*>(base);
    base += sizeof(PipelineQ::SharedStorage);
    base = (base + 15) & ~uintptr_t(15);
    float* sLSE_local = reinterpret_cast<float*>(base);

    // CuTe tensor views (used by WG1)
    Tensor sQ0 = make_tensor(make_smem_ptr(sQ0_ptr), SmemLayoutQ{});
    Tensor sQ1 = make_tensor(make_smem_ptr(sQ1_ptr), SmemLayoutQ{});
    Tensor sK  = make_tensor(make_smem_ptr(sK_ptr),  SmemLayoutK{});
    Tensor sV  = make_tensor(make_smem_ptr(sV_ptr),  SmemLayoutV{});
    Tensor sO0 = make_tensor(make_smem_ptr(sO0_ptr), SmemLayoutO{});
    Tensor sO1 = make_tensor(make_smem_ptr(sO1_ptr), SmemLayoutO{});

    // sX_p[stage]: X staging buffers alias sQ0/sQ1 during projection subphase
    Element* sX_p[2] = { sQ0_ptr, sQ1_ptr };

    // TMA byte constants
    constexpr int kTmaXBytes   = kBlockN  * kHiddenChunk * (int)sizeof(Element);
    constexpr int kTmaWBytes   = kHeadDim * kHiddenChunk * (int)sizeof(Element);
    constexpr int kTmaQHalf    = kBlockM  * 64            * (int)sizeof(Element);

    // Chunk stride within W_full smem (in elements)
    constexpr int kChunkStrideW = cute::cosize(SmemLayoutW{});

    // ------------------------------------------------------------------
    // Pipeline construction (all 256 threads must participate)
    // ------------------------------------------------------------------
    PipelineProj::Params proj_p;
    proj_p.role = (wg_idx == 0) ? PipelineProj::ThreadCategory::Producer
                                 : PipelineProj::ThreadCategory::Consumer;
    proj_p.transaction_bytes = static_cast<uint32_t>(kTmaXBytes);  // 8KB — X only
    proj_p.is_leader         = (wg_idx == 0 && tid == 0) ? 1u : 0u;
    proj_p.num_consumers     = static_cast<uint32_t>(kNThreadsMMA);
    PipelineProj pipeline_proj(*pipe_proj_smem, proj_p, Shape<_1,_1,_1>{});

    PipelineQ::Params q_p;
    q_p.role = (wg_idx == 0) ? PipelineQ::ThreadCategory::Producer
                              : PipelineQ::ThreadCategory::Consumer;
    q_p.transaction_bytes = 2u * static_cast<uint32_t>(kTmaQHalf);  // 16KB — two halves
    q_p.is_leader         = (wg_idx == 0 && tid == 0) ? 1u : 0u;
    q_p.num_consumers     = static_cast<uint32_t>(kNThreadsMMA);
    PipelineQ pipeline_q(*pipe_q_smem, q_p, Shape<_1,_1,_1>{});

    // Startup mbarrier init (tid==0 only; __syncthreads ensures visibility)
    if (tid == 0) {
        mbar_init(startup_mbar, 1);
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // STARTUP: TMA-load Wk_full and Wv_full for this head into persistent smem
    // All threads wait on startup_mbar before the outer loop.
    // ------------------------------------------------------------------
    {
        constexpr int kStartupBytes = kNumProjChunks * kTmaWBytes * 2;  // 64KB

        if (tid == 0) {
            uint32_t mbar_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(startup_mbar));
            mbar_arrive_expect_tx(startup_mbar, kStartupBytes);
            uint32_t wk_base = static_cast<uint32_t>(__cvta_generic_to_shared(sWk_ptr));
            uint32_t wv_base = static_cast<uint32_t>(__cvta_generic_to_shared(sWv_ptr));
            for (int cs = 0; cs < kNumProjChunks; ++cs) {
                int h_off = cs * kHiddenChunk;
                uint32_t wk_addr = wk_base + cs * kChunkStrideW * sizeof(Element);
                uint32_t wv_addr = wv_base + cs * kChunkStrideW * sizeof(Element);
                tma_load_3d(wk_addr, params.tma_Wk, h_off, 0, head, mbar_addr);
                tma_load_3d(wv_addr, params.tma_Wv, h_off, 0, head, mbar_addr);
            }
        }
        // All 256 threads spin until Wk_full/Wv_full are fully in smem.
        // fence.proxy.async makes the cp.async.bulk data visible to WGMMA (SM90 requirement).
        mbar_wait(startup_mbar, 0);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();   // memory fence: sWk_full/sWv_full visible to all threads
        if (dbg_wg0) atomicAdd(&params.debug_ctrs[0], 1u);  // CP0: past startup mbar
    }

    // ------------------------------------------------------------------
    // Pipeline states declared BEFORE outer loop — must advance monotonically
    // across all n_block iterations (phase continuity).
    // ------------------------------------------------------------------
    auto pipe_proj_write = cutlass::make_producer_start_state<PipelineProj>();
    cutlass::PipelineState<2> pipe_proj_read{};

    auto pipe_q_write = cutlass::make_producer_start_state<PipelineQ>();
    cutlass::PipelineState<2> pipe_q_read{};

    const float softmax_scale_log2 = params.softmax_scale * float(M_LOG2E);
    constexpr int kNRows = 2 * (kBlockM / 64);

    // ------------------------------------------------------------------
    // Outer loop: for each KV tile (n_block)
    // ------------------------------------------------------------------
    for (int n_block = 0; n_block < params.num_n_blocks; ++n_block) {
        const int n_start = n_block * kBlockN;

        // ----------------------------------------------------------------
        // PROJECTION SUBPHASE
        // WG0: TMA producer — loads X chunks into sX_p[stage]
        // WG1: WGMMA consumer — X×sWk → acc_k, X×sWv → acc_v; writes sK, sV
        // ----------------------------------------------------------------
        if (wg_idx == 0) {
            // Only tid==0 participates in the barrier protocol and issues TMA.
            // Threads 1..127 have no barrier role; if they called producer_acquire they
            // could race: WG1 may signal empty_barrier[stage] (releasing it to phase=1)
            // before a delayed WG0 warp reaches producer_acquire, causing a permanent hang.
            for (int cs = 0; cs < kNumProjChunks; ++cs) {
                if (tid == 0) {
                    pipeline_proj.producer_acquire(pipe_proj_write);
                    if (dbg_wg0) atomicAdd(&params.debug_ctrs[1 + cs], 1u);  // CP1,CP2
                    int h_off = cs * kHiddenChunk;
                    int stg   = pipe_proj_write.index();
                    uint32_t bar_addr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(
                            pipeline_proj.producer_get_barrier(pipe_proj_write)));
                    uint32_t sx_addr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(sX_p[stg]));
                    tma_load_3d(sx_addr, params.tma_X, h_off, n_start, batch, bar_addr);
                }
                ++pipe_proj_write;
            }
        } else {
            // WG1: accumulate acc_k = X[n_block] @ Wk[head].T
            //                acc_v = X[n_block] @ Wv[head].T
            TiledMmaProj tiled_mma_proj;
            auto thr = tiled_mma_proj.get_thread_slice(ctid);
            Tensor acc_k = partition_fragment_C(
                tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
            Tensor acc_v = partition_fragment_C(
                tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
            clear(acc_k); clear(acc_v);

            for (int cs = 0; cs < kNumProjChunks; ++cs) {
                pipeline_proj.consumer_wait(pipe_proj_read);
                if (dbg_wg1) atomicAdd(&params.debug_ctrs[7 + cs], 1u);  // CP7,CP8: proj wait
                int stg = pipe_proj_read.index();

                // X from pipeline buffer; Wk/Wv from persistent smem (chunk cs)
                Tensor sX_cur  = make_tensor(make_smem_ptr(sX_p[stg]), SmemLayoutX{});
                Tensor sWk_cur = make_tensor(
                    make_smem_ptr(sWk_ptr + cs * kChunkStrideW), SmemLayoutW{});
                Tensor sWv_cur = make_tensor(
                    make_smem_ptr(sWv_ptr + cs * kChunkStrideW), SmemLayoutW{});

                auto tA  = thr.partition_fragment_A(sX_cur);
                auto tBk = thr.partition_fragment_B(sWk_cur);
                auto tBv = thr.partition_fragment_B(sWv_cur);

                if (cs == 0) {
                    flash::gemm<true,  -1>(tiled_mma_proj, tA, tBk, acc_k);
                    flash::gemm<true,  -1>(tiled_mma_proj, tA, tBv, acc_v);
                } else {
                    flash::gemm<false, -1>(tiled_mma_proj, tA, tBk, acc_k);
                    flash::gemm<false, -1>(tiled_mma_proj, tA, tBv, acc_v);
                }
                cute::warpgroup_wait<0>();
                if (cs == kNumProjChunks - 1 && dbg_wg1)
                    atomicAdd(&params.debug_ctrs[12], 1u);  // CP12: wgwait done cs=last
                cute::warpgroup_fence_operand(acc_k);
                cute::warpgroup_fence_operand(acc_v);

                pipeline_proj.consumer_release(pipe_proj_read);
                ++pipe_proj_read;
            }
            if (dbg_wg1) atomicAdd(&params.debug_ctrs[13], 1u);  // CP13: after proj loop

            // Store acc_k → sK (K-major), acc_v → sV (MN-major transposed)
            Tensor tCsK = thr.partition_C(sK);
            Tensor cV   = cute::make_identity_tensor(
                Shape<Int<kBlockN>, Int<kHeadDim>>{});
            Tensor tCcV = thr.partition_C(cV);
            #pragma unroll
            for (int i = 0; i < size(acc_k); ++i) {
                tCsK(i) = Element(acc_k(i));
                sV(get<1>(tCcV(i)), get<0>(tCcV(i))) = Element(acc_v(i));
            }
        }

        // SYNC A: sK/sV ready; sQ0/sQ1 (X staging) free for Q use
        __syncthreads();
        if (dbg_wg0) atomicAdd(&params.debug_ctrs[3], 1u);  // CP3: past SYNC A

        // ----------------------------------------------------------------
        // ATTENTION SUBPHASE
        // Causal: Q tile at m_block can attend to this KV tile if
        //   m_start <= n_start + kBlockN - 1
        // WG0: TMA producer — loads Q tiles (double-buffered sQ0/sQ1)
        // WG1: attention consumer — QK, softmax, PV, STSM, LSE merge
        // ----------------------------------------------------------------
        const int m_end = params.is_causal
            ? min(params.num_m_blocks,
                  (n_start + kBlockN + kBlockM - 1) / kBlockM)
            : params.num_m_blocks;

        if (wg_idx == 0) {
            // WG0: Q TMA producer — only tid==0 calls producer_acquire.
            // Same race-fix as the proj loop: non-leader WG0 threads must not call
            // producer_acquire, or they can deadlock when WG1's consumer_release advances
            // the empty_barrier past the phase a late warp is waiting for.
            for (int m_block = 0; m_block < m_end; ++m_block) {
                if (tid == 0) {
                    pipeline_q.producer_acquire(pipe_q_write);
                    if (dbg_wg0) atomicAdd(&params.debug_ctrs[4], 1u);  // CP4: Q acquire
                    int m_start = m_block * kBlockM;
                    int stg = pipe_q_write.index();
                    Element* sq = (stg == 0) ? sQ0_ptr : sQ1_ptr;
                    uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(sq));
                    uint32_t s1 = s0 + kTmaQHalf;
                    uint32_t mb = static_cast<uint32_t>(
                        __cvta_generic_to_shared(
                            pipeline_q.producer_get_barrier(pipe_q_write)));
                    tma_load_4d(s0, params.tma_Q, 0,  head, m_start, batch, mb);
                    tma_load_4d(s1, params.tma_Q, 64, head, m_start, batch, mb);
                }
                ++pipe_q_write;
            }
        } else {
            // WG1: attention consumer
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

            for (int m_block = 0; m_block < m_end; ++m_block) {
                const int m_start = m_block * kBlockM;
                pipeline_q.consumer_wait(pipe_q_read);
                if (dbg_wg1) atomicAdd(&params.debug_ctrs[9], 1u);  // CP9: Q wait
                int buf = pipe_q_read.index();

                // QK GEMM
                Tensor acc_s = partition_fragment_C(
                    tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
                if (buf == 0) {
                    flash::gemm<true, -1>(tiled_mma_qk, tSrQ0, tSrK, acc_s);
                } else {
                    flash::gemm<true, -1>(tiled_mma_qk, tSrQ1, tSrK, acc_s);
                }
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(acc_s);

                // Release Q buffer early — WG0 can load next Q while we do softmax+PV
                pipeline_q.consumer_release(pipe_q_read);
                ++pipe_q_read;

                // Causal mask
                if (params.is_causal) {
                    auto thread0_mma = TiledMmaQK{}.get_thread_slice(_0{});
                    Tensor cS    = cute::make_identity_tensor(
                        Shape<Int<kBlockM>, Int<kBlockN>>{});
                    Tensor tScS  = thr_mma_qk.partition_C(cS);
                    Tensor t0ScS = thread0_mma.partition_C(cS);
                    Tensor acc_s_rc = make_tensor(acc_s.data(),
                        flash::convert_layout_acc_rowcol(acc_s.layout()));
                    Tensor tScS_rc  = make_tensor(tScS.data(),
                        flash::convert_layout_acc_rowcol(tScS.layout()));
                    Tensor t0ScS_rc = make_tensor(t0ScS.data(),
                        flash::convert_layout_acc_rowcol(t0ScS.layout()));
                    int thread_col_offset = get<1>(tScS_rc(_0{}, _0{}));
                    int causal_row_offset = m_start + 1 - n_start - thread_col_offset;
                    #pragma unroll
                    for (int m = 0; m < size<0>(acc_s_rc); ++m) {
                        int row_rel   = get<0>(tScS_rc(m, _0{}));
                        int col_limit = row_rel + causal_row_offset;
                        #pragma unroll
                        for (int n = 0; n < size<1>(acc_s_rc); ++n) {
                            int col_rel_t0 = get<1>(t0ScS_rc(_0{}, n));
                            if (col_rel_t0 >= col_limit) { acc_s_rc(m, n) = -INFINITY; }
                        }
                    }
                }

                // Softmax (standalone per KV tile: first=true for this m_block's view)
                flash::Softmax<kNRows, 0> softmax(softmax_scale_log2);
                (void)softmax.template max_get_scale<true, true>(acc_s);
                softmax.template online_softmax<true, true>(acc_s);

                // PV GEMM (RS-mode: P in regs, V in smem)
                Tensor tOrP_acc = make_tensor(acc_s.data(),
                    flash::convert_layout_acc_Aregs<TiledMmaPV>(acc_s.layout()));
                Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
                flash::convert_type_out(tOrP_acc, tOrP);

                Tensor acc_o = partition_fragment_C(
                    tiled_mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});
                flash::gemm<true, -1>(tiled_mma_pv, tOrP, tOrV, acc_o);

                // Finalize softmax: reduce row_sum → LSE; get 1/l_local scale
                auto scores_scale = softmax.finalize();
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(acc_o);
                softmax.rescale_o(acc_o, scores_scale);

                // (a) Write LSE_local per row to sLSE_local (lane % 4 == 0)
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

                // (b) STSM acc_o → sO (ping-pong on m_block % 2)
                {
                    Tensor rO = make_tensor_like<Element>(acc_o);
                    flash::convert_type_out(acc_o, rO);
                    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
                    if ((m_block & 1) == 0) {
                        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO0);
                    } else {
                        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO1);
                    }
                }

                // Sync: sO and sLSE_local ready for all 128 WG1 threads
                cutlass::arch::NamedBarrier::sync(
                    static_cast<uint32_t>(kNThreadsMMA),
                    static_cast<uint32_t>(SmpV2Barrier::OSmemReady));
                if (dbg_wg1) atomicAdd(&params.debug_ctrs[10], 1u);  // CP10: OSmemReady

                // (c) Lock-free LSE-form merge into HBM O_run/LSE_run
                //     This CTA owns all rows for (batch, head) → no spinlock needed.
                //     Thread assignment: ctid>>1 = row_in_tile, ctid&1 = d_half
                {
                    Element* sO_cur_p = ((m_block & 1) == 0) ? sO0_ptr : sO1_ptr;
                    Tensor sO_cur = make_tensor(make_smem_ptr(sO_cur_p), SmemLayoutO{});
                    int row_in_tile = ctid >> 1;
                    int d_half      = ctid & 1;
                    int s_global    = m_start + row_in_tile;

                    if (s_global < S_q) {
                        int row_id = (batch * S_q + s_global) * NH + head;

                        float LSE_local_val = sLSE_local[row_in_tile];
                        float LSE_old = LSE_run_ptr[row_id];

                        float alpha, beta, LSE_new;
                        float LSE_max = fmaxf(LSE_old, LSE_local_val);
                        if (LSE_max == -INFINITY) {
                            alpha = 0.f; beta = 0.f; LSE_new = -INFINITY;
                        } else {
                            float a_old = (LSE_old       == -INFINITY) ? 0.f
                                : __expf(LSE_old       - LSE_max);
                            float a_loc = (LSE_local_val == -INFINITY) ? 0.f
                                : __expf(LSE_local_val - LSE_max);
                            float Z   = a_old + a_loc;
                            LSE_new   = LSE_max + __logf(Z);
                            alpha = (LSE_old       == -INFINITY) ? 0.f
                                : __expf(LSE_old       - LSE_new);
                            beta  = (LSE_local_val == -INFINITY) ? 0.f
                                : __expf(LSE_local_val - LSE_new);
                        }

                        constexpr int kHalfD = kHeadDim / 2;
                        int d_start = d_half * kHalfD;
                        #pragma unroll
                        for (int d = 0; d < kHalfD; d += 4) {
                            int d_global = d_start + d;
                            float4 o_old = *reinterpret_cast<const float4*>(
                                &O_run_ptr[row_id * kHeadDim + d_global]);
                            Element so0_ = sO_cur(row_in_tile, d_global + 0);
                            Element so1_ = sO_cur(row_in_tile, d_global + 1);
                            Element so2_ = sO_cur(row_in_tile, d_global + 2);
                            Element so3_ = sO_cur(row_in_tile, d_global + 3);
                            float4 o_new;
                            o_new.x = alpha * o_old.x + beta * (float)so0_;
                            o_new.y = alpha * o_old.y + beta * (float)so1_;
                            o_new.z = alpha * o_old.z + beta * (float)so2_;
                            o_new.w = alpha * o_old.w + beta * (float)so3_;
                            *reinterpret_cast<float4*>(
                                &O_run_ptr[row_id * kHeadDim + d_global]) = o_new;
                        }
                        if (d_half == 0) {
                            LSE_run_ptr[row_id] = LSE_new;
                        }
                    }

                    cutlass::arch::NamedBarrier::sync(
                        static_cast<uint32_t>(kNThreadsMMA),
                        static_cast<uint32_t>(SmpV2Barrier::MergeStatsReady));
                    if (dbg_wg1) atomicAdd(&params.debug_ctrs[11], 1u);  // CP11: MergeStatsReady
                }
            }   // end m_block loop
        }       // end wg_idx == 1 attention block

        // SYNC B: attention done; next n_block's X staging is safe
        __syncthreads();
        if (dbg_wg0) atomicAdd(&params.debug_ctrs[5], 1u);  // CP5: past SYNC B

    }   // end n_block loop

    // ------------------------------------------------------------------
    // FINALIZE: cast O_run (fp32) → O (bf16)
    // Unconditional: this CTA owns the entire (batch, head) slice.
    // ------------------------------------------------------------------
    __syncthreads();

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
size_t sm_parallel_v2_smem_bytes()
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
    size_t smem = 128;  // base alignment sentinel
    smem += r128(cute::cosize(typename Traits::SmemLayoutQ{}) * sizeof(Element));  // sQ0
    smem += r128(cute::cosize(typename Traits::SmemLayoutQ{}) * sizeof(Element));  // sQ1
    smem += r128(cute::cosize(SmemLayoutW_full{}) * sizeof(Element));              // sWk_full
    smem += r128(cute::cosize(SmemLayoutW_full{}) * sizeof(Element));              // sWv_full
    smem += r128(cute::cosize(typename Traits::SmemLayoutK{}) * sizeof(Element));  // sK
    smem += r128(cute::cosize(typename Traits::SmemLayoutV{}) * sizeof(Element));  // sV
    smem += r128(cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));  // sO0
    smem += r128(cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));  // sO1
    smem += 8;    // startup_mbar
    smem += sizeof(PipelineProj::SharedStorage) + 16;
    smem += sizeof(PipelineQ::SharedStorage)    + 16;
    smem += Traits::kBlockM * sizeof(float) + 16;   // sLSE_local
    return smem;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
template <typename Traits>
cudaError_t launch_sm_parallel_v2(SmParallelV2Params p, cudaStream_t stream = 0)
{
    size_t smem = sm_parallel_v2_smem_bytes<Traits>();
    auto* kernel = &sm_parallel_v2_fwd_kernel<Traits>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (err != cudaSuccess) return err;
    dim3 grid(p.num_heads, p.batch);
    kernel<<<grid, 256, smem, stream>>>(p);
    return cudaGetLastError();
}

} // namespace sm_parallel_v2
