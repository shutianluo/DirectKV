/******************************************************************************
 * Phase 3 — Fused Projection + Attention Kernel (SM90)
 *
 * Synchronization model follows Hopper FA3 exactly:
 *   PipelineTmaAsync<2>  for projection (Phase 1) and Q loading (Phase 2)
 *   PipelineState<2>     for circular buffer tracking
 *   NamedBarrier         for intra-WG1 STSM sync
 *
 * No atomicAdd spin-poll, no smem_done flags, no hand-rolled mbarrier arrays,
 * no unnamed bar.sync, no manual PipeState.
 *
 * Thread organization (256 threads / 2 warpgroups):
 *
 * Phase 1 (projection or past-KV load):
 *   WG0: TMA producer for X, Wk, Wv  (tid=0 is_leader; all 128 call producer_acquire)
 *   WG1: WGMMA consumer for K/V projection
 *   Both: uint4 stores sK → Kc, sV → Vc  (new-KV path only)
 *
 * Phase boundary: __syncthreads() — sK/sV ready, projection temporaries dead
 *
 * Phase 2 (attention loop):
 *   WG0: TMA Q producer  (tid=0 is_leader; all 128 call producer_acquire; then return)
 *   WG1: QK WGMMA → softmax → PV WGMMA → STSM → uint4 Opart store
 *
 * Smem layout (aliased, ~96 KB):
 *   sQ0(=sX alias) sQ1 sK sV sO0(=sWk alias) sO1(=sWv alias) + barrier region
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
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include "proj_fused_kernel_traits_sm90.h"
#include "softmax.h"
#include "utils.h"
#include <cutlass/arch/reg_reconfig.h>

namespace fused_v3 {

using namespace cute;

// ---------------------------------------------------------------------------
// TMA helper — 3D bulk tensor load (descriptor-based, mbarrier-signaled)
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

// ---------------------------------------------------------------------------
// Named barrier enum for intra-WG1 synchronization
// ---------------------------------------------------------------------------

enum class FusedBarrier : uint32_t { OSmemReady = 0 };

// ---------------------------------------------------------------------------
// CUTLASS pipeline type aliases
// ---------------------------------------------------------------------------

using PipelineProj = cutlass::PipelineTmaAsync<2>;
using PipelineQ    = cutlass::PipelineTmaAsync<2>;

// ---------------------------------------------------------------------------
// Param struct
// ---------------------------------------------------------------------------
struct FusedParams {
    void const* __restrict__ ptr_X;
    void const* __restrict__ ptr_Wk;
    void const* __restrict__ ptr_Wv;
    void const* __restrict__ ptr_Q;
    void*       __restrict__ ptr_Opart;
    void*       __restrict__ ptr_Kc;
    void*       __restrict__ ptr_Vc;
    float*      __restrict__ ptr_LSE;
    void const* __restrict__ ptr_Kpast;
    void const* __restrict__ ptr_Vpast;
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
// Kernel
// ---------------------------------------------------------------------------

template <typename Traits>
__global__ __launch_bounds__(256, 1)
void fused_fwd_kernel(FusedParams params)
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

    const int n_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int tid     = threadIdx.x;

    const int wg_idx = tid / 128;
    const int ctid   = tid - 128;   // WG1-local [0,127]

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

    auto* Op_ptr  = reinterpret_cast<Element*>(params.ptr_Opart);
    auto* Kc_ptr  = reinterpret_cast<Element*>(params.ptr_Kc);
    auto* Vc_ptr  = reinterpret_cast<Element*>(params.ptr_Vc);
    float* LSE_ptr = params.ptr_LSE;
    auto const* Kpast_ptr = reinterpret_cast<Element const*>(params.ptr_Kpast);
    auto const* Vpast_ptr = reinterpret_cast<Element const*>(params.ptr_Vpast);

    // ------------------------------------------------------------------
    // Shared memory — aliased layout (~96 KB):
    //   sQ0  [kBlockM×kHeadDim]  16 KB   Q ping / alias sX (Phase 1)
    //   sQ1  [kBlockM×kHeadDim]  16 KB   Q pong
    //   sK   [kBlockN×kHeadDim]  16 KB   K output (Phase 1) / K input (Phase 2)
    //   sV   [kHeadDim×kBlockN]  16 KB   V output / input
    //   sO0  [kBlockM×kHeadDim]  16 KB   O ping / alias sWk (Phase 1)
    //   sO1  [kBlockM×kHeadDim]  16 KB   O pong / alias sWv (Phase 1)
    //   barrier region           256 B   PipelineProj::SS + PipelineQ::SS + alignment
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

    // Barrier region: place the two pipeline SharedStorage objects.
    // 16-byte alignment required by CUTLASS mbarrier.
    uintptr_t bar_base = align128(reinterpret_cast<uintptr_t>(sO1_ptr)
                                  + cute::cosize(SmemLayoutO{}) * sizeof(Element));
    bar_base = (bar_base + 15) & ~uintptr_t(15);
    auto* pipe_proj_smem = reinterpret_cast<PipelineProj::SharedStorage*>(bar_base);
    bar_base += sizeof(PipelineProj::SharedStorage);
    bar_base  = (bar_base + 15) & ~uintptr_t(15);
    auto* pipe_q_smem    = reinterpret_cast<PipelineQ::SharedStorage*>(bar_base);

    // ------------------------------------------------------------------
    // TMA byte-count constants (compile-time)
    // ------------------------------------------------------------------
    constexpr int kTmaXBytes    = kBlockN  * kHiddenChunk * (int)sizeof(Element);
    constexpr int kTmaWBytes    = kHeadDim * kHiddenChunk * (int)sizeof(Element);
    constexpr int kTmaProjTotal = kTmaXBytes + 2 * kTmaWBytes;
    constexpr int kTmaHalfBytes = kBlockM * 64 * (int)sizeof(Element);  // Q split into two halves

    // ------------------------------------------------------------------
    // Pipeline construction — all 256 threads participate.
    //
    // Each thread creates a thread-local PipelineProj / PipelineQ object.
    // The constructors:
    //   1. Warp 0 initializes the shared-memory barriers (full + empty arrays).
    //   2. All threads call fence_barrier_init() (release fence).
    // One explicit __syncthreads() acquires those barrier initializations.
    // ------------------------------------------------------------------
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

    __syncthreads();  // acquire visibility of initialized barriers across all threads

    // ==================================================================
    // PHASE 1: K/V computation — either load past KV or project new KV
    // ==================================================================

    if (is_past_kv) {
        // Synchronous ld.global uint4 with inline V-transpose
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
        // -----------------------------------------------------------------
        // New-KV path: FA3-style TMA 3D double-buffered projection pipeline
        //
        // Double-buffer stage aliases (reuse existing smem, no extra allocation):
        //   Stage 0: sX → sQ0_ptr, sWk → sO0_ptr, sWv → sO1_ptr
        //   Stage 1: sX → sQ1_ptr, sWk → sK_ptr,  sWv → sV_ptr
        //
        // WG0 (producer): all 128 threads call producer_acquire.
        //   - is_leader (tid==0): arrive_and_expect_tx + TMA loads.
        //   - Others: wait on empty barrier, then do nothing.
        // WG1 (consumer): all 128 threads call consumer_wait/release.
        // -----------------------------------------------------------------
        Element* sX_p[2]  = { sQ0_ptr, sQ1_ptr };
        Element* sWk_p[2] = { sO0_ptr, sK_ptr  };
        Element* sWv_p[2] = { sO1_ptr, sV_ptr  };

        if (wg_idx == 0) {
            // ---- WG0: TMA PRODUCER ----
            auto smem_pipe_write = cutlass::make_producer_start_state<PipelineProj>();

            for (int cs = 0; cs < kNumProjChunks; ++cs) {
                // All 128 WG0 threads wait on empty barrier.
                // tid==0 (is_leader) also calls arrive_and_expect_tx on full barrier.
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
            // Fall through to __syncthreads() + K/V store below
        } else {
            // ---- WG1: WGMMA CONSUMER ----
            TiledMmaProj tiled_mma_proj;
            auto thr_mma = tiled_mma_proj.get_thread_slice(ctid);
            Tensor acc_k = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
            Tensor acc_v = partition_fragment_C(tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
            clear(acc_k); clear(acc_v);

            auto smem_pipe_read = cutlass::PipelineState<2>{};  // {idx=0, phase=0}

            for (int cs = 0; cs < kNumProjChunks; ++cs) {
                // All 128 WG1 threads spin on full barrier until TMA completes.
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

                // All 128 WG1 threads arrive on empty barrier.
                // When all 128 have arrived the barrier opens → WG0 producer_acquire unblocks.
                pipeline_proj.consumer_release(smem_pipe_read);
                ++smem_pipe_read;
            }

            // Write WGMMA accumulators → sK (K-major) and sV (MN-major)
            Tensor tCsK = thr_mma.partition_C(sK);
            Tensor cV   = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
            Tensor tCcV = thr_mma.partition_C(cV);
            #pragma unroll
            for (int i = 0; i < size(acc_k); ++i) {
                tCsK(i) = Element(acc_k(i));
                sV(get<1>(tCcV(i)), get<0>(tCcV(i))) = Element(acc_v(i));
            }
        }
        // Sync WG0 + WG1: WG1's sK/sV writes must be visible before the K/V store.
        __syncthreads();

        // Store sK → Kc, sV → Vc (all 256 threads, uint4)
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

    // ==================================================================
    // Phase boundary: sK/sV stable; projection temporaries dead
    // ==================================================================
    __syncthreads();

    const int new_n_idx     = n_block - num_n_blocks_past;
    const int m_start_block = (params.is_causal && !is_past_kv && new_n_idx > 0)
                              ? new_n_idx : 0;

    // ==================================================================
    // PHASE 2: attention — WG0 = TMA producer, WG1 = consumer
    // ==================================================================

    if (wg_idx == 0) {
        // ---- WG0: Q TMA PRODUCER ----
        // All 128 WG0 threads call producer_acquire (waits on empty barrier).
        // Only tid==0 (is_leader) calls arrive_and_expect_tx + issues TMA.
        // After issuing all Q tiles, WG0 returns.
        CUtensorMap const* tma_desc = params.tma_desc_Q;

        auto smem_pipe_write_q = cutlass::make_producer_start_state<PipelineQ>();

        for (int m_block = m_start_block; m_block < params.num_m_blocks; ++m_block) {
            // Waits for empty barrier: WG1's 128 consumer_release arrives opened it.
            // First two iterations: empty barriers are pre-initialized open (phase 1).
            pipeline_q.producer_acquire(smem_pipe_write_q);

            if (tid == 0) {
                int m_start = m_block * kBlockM;
                int idx = smem_pipe_write_q.index();
                Element* sq = (idx == 0) ? sQ0_ptr : sQ1_ptr;
                uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(sq));
                uint32_t s1 = s0 + kTmaHalfBytes;
                uint32_t mb = static_cast<uint32_t>(
                    __cvta_generic_to_shared(pipeline_q.producer_get_barrier(smem_pipe_write_q)));
                // Q is split into two 64-element-wide 4D TMA slices
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
        return;
    }

    // ---- CONSUMER (WG1) ----
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
    const int num_n_blocks_total = params.num_n_blocks;

    // Write LSE=-INF for causal-skipped m_blocks
    for (int m_block = 0; m_block < m_start_block; ++m_block) {
        int m_start = m_block * kBlockM;
        for (int i = ctid; i < kBlockM; i += kNThreadsMMA) {
            int row = m_start + i;
            size_t off_lse = (((size_t)batch * S + row) * NH + head)
                           * num_n_blocks_total + n_block;
            LSE_ptr[off_lse] = -INFINITY;
        }
    }

    // Q consumer pipeline state
    auto smem_pipe_read_q = cutlass::PipelineState<2>{};  // {idx=0, phase=0}

    for (int m_block = m_start_block; m_block < params.num_m_blocks; ++m_block) {
        int m_start = m_block * kBlockM;

        // Capture buffer index before pipeline advance (used for both Q and O selection)
        int o_buf = smem_pipe_read_q.index();

        // All 128 WG1 threads spin on full barrier until TMA completes.
        // wgmma.fence.sync.aligned (inside flash::gemm<Sync=true>) provides the
        // collective warpgroup sync before the first WGMMA — no separate bar needed.
        pipeline_q.consumer_wait(smem_pipe_read_q);

        Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
        if (o_buf == 0) {
            flash::gemm<true, -1>(tiled_mma_qk, tSrQ0, tSrK, acc_s);
        } else {
            flash::gemm<true, -1>(tiled_mma_qk, tSrQ1, tSrK, acc_s);
        }
        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(acc_s);

        // Signal Q smem free: all 128 WG1 threads arrive on empty barrier.
        // WG0's producer_acquire for the next Q tile unblocks when all 128 arrive.
        pipeline_q.consumer_release(smem_pipe_read_q);
        ++smem_pipe_read_q;

        // Overlap: store previous m_block's O from sO[1-o_buf]
        if (m_block > m_start_block) {
            auto& sO_prev = (o_buf == 0) ? sO1 : sO0;
            int m_prev_start = m_start - kBlockM;
            #pragma unroll
            for (int v = 0; v < kVecsPerThrO; ++v) {
                int idx = v * kNThreadsMMA + ctid;
                int i = idx / kVecsPerRowO;
                int c = (idx % kVecsPerRowO) * kElemsPerVecO;
                int row = m_prev_start + i;
                size_t off = ((((size_t)batch * S + row) * NH + head)
                           * num_n_blocks_total + n_block) * kHeadDim + c;
                uint4 val = *reinterpret_cast<uint4 const*>(&sO_prev(i, c));
                *reinterpret_cast<uint4*>(Op_ptr + off) = val;
            }
        }

        // Causal mask
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
                    size_t off_lse = (((size_t)batch * S + row) * NH + head)
                                   * num_n_blocks_total + n_block;
                    LSE_ptr[off_lse] = lse;
                }
            }
        }

        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(acc_o);
        softmax.rescale_o(acc_o, scores_scale);

        // STSM O to smem[o_buf], then sync WG1 so sO is ready for next iteration's Opart read.
        {
            Tensor rO = make_tensor_like<Element>(acc_o);
            flash::convert_type_out(acc_o, rO);
            Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
            auto& taccOsO_cur = (o_buf == 0) ? taccOsO0 : taccOsO1;
            cute::copy(smem_tiled_copy_O, taccOrO, taccOsO_cur);
            // All 128 WG1 threads: barrier after STSM ensures sO writes are globally
            // visible before the next iteration reads sO_prev for Opart store.
            cutlass::arch::NamedBarrier::sync(
                static_cast<uint32_t>(kNThreadsMMA),
                static_cast<uint32_t>(FusedBarrier::OSmemReady));
        }
    }

    // Post-loop: store the last m_block's O.
    // Buffer index = (num_m_blocks - 1 - m_start_block) & 1  (matches pipeline state at loop end)
    if (m_start_block < params.num_m_blocks) {
        int last_m_start = (params.num_m_blocks - 1) * kBlockM;
        int last_o_buf   = (params.num_m_blocks - 1 - m_start_block) & 1;
        auto& sO_last    = (last_o_buf == 0) ? sO0 : sO1;
        #pragma unroll
        for (int v = 0; v < kVecsPerThrO; ++v) {
            int idx = v * kNThreadsMMA + ctid;
            int i   = idx / kVecsPerRowO;
            int c   = (idx % kVecsPerRowO) * kElemsPerVecO;
            int row = last_m_start + i;
            size_t off = ((((size_t)batch * S + row) * NH + head)
                       * num_n_blocks_total + n_block) * kHeadDim + c;
            uint4 val = *reinterpret_cast<uint4 const*>(&sO_last(i, c));
            *reinterpret_cast<uint4*>(Op_ptr + off) = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------
template <typename Traits>
cudaError_t launch_fused(
    FusedParams p,
    cudaStream_t stream = 0)
{
    dim3 grid(p.num_n_blocks, p.num_heads, p.batch);
    dim3 block(256);

    using Element = typename Traits::Element;
    auto r128 = [](size_t s) { return (s + 127) & ~size_t(127); };
    size_t smem = 128;
    smem += r128(cute::cosize(typename Traits::SmemLayoutQ{}) * sizeof(Element)) * 2; // sQ0, sQ1
    smem += r128(cute::cosize(typename Traits::SmemLayoutK{}) * sizeof(Element));      // sK
    smem += r128(cute::cosize(typename Traits::SmemLayoutV{}) * sizeof(Element));      // sV
    smem += r128(cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element)) * 2; // sO0, sO1
    smem += 256;  // barrier region: 2x PipelineTmaAsync<2>::SharedStorage + alignment

    auto* kernel = &fused_fwd_kernel<Traits>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (err != cudaSuccess) return err;
    kernel<<<grid, block, smem, stream>>>(p);
    return cudaGetLastError();
}

} // namespace fused_v3
