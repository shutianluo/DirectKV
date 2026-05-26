/******************************************************************************
 * Projection-Fused Flash Attention — Full FA3-style Kernel (SM90)
 *
 * All data movement via TMA. WG0 = producer (TMA loads), WG1 = consumer (WGMMA).
 *
 * Phase 1 — Projection or Past-KV fetch:
 *   Projection path: WG0 warp 0 TMA-loads X/Wk/Wv chunks into double-buffered
 *     smem (kStages=2), WG1 consumes via WGMMA. Pipeline uses proj_full/proj_empty
 *     mbarriers. After WGMMA: results written to sK/sV. V transposed to sQ0 staging,
 *     then TMA-stored to Kc/Vc in HBM.
 *   Past-KV path: WG0 warp 0 TMA-loads Kpast directly into sK, Vpast into sQ0
 *     staging (row-major). All threads transpose V into sV (MN-major for PV GEMM).
 *
 * Phase 2 — Attention:
 *   WG0 warp 0 TMA-loads Q tiles (double-buffered sQ0/sQ1, q_full/q_empty mbarriers).
 *   WG1 computes QK (SS WGMMA), softmax, PV (RS WGMMA), STSM O to smem.
 *   O tiles TMA-stored to Opart in HBM (overlapped with next QK).
 *
 * Grid: (num_n_blocks, num_heads, batch).  256 threads: WG0 (128) + WG1 (128).
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
#include <cutlass/numeric_conversion.h>
#include <cutlass/array.h>

#include "proj_fused_kernel_traits_sm90.h"
#include "softmax.h"
#include "utils.h"

namespace proj_fused {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////
// Params — TMA descriptors for all data movement, scalar ptr for LSE only
////////////////////////////////////////////////////////////////////////////////
struct WarpSpecParams {
    // LSE: scalar per-thread writes (not TMA-suitable)
    float*      __restrict__ ptr_LSE;
    // Dimensions
    int batch, seqlen, num_heads, hidden_dim, head_dim;
    int seqlen_past, num_n_blocks_past, num_n_blocks, num_m_blocks;
    float softmax_scale;
    int is_causal;
    // TMA descriptors (device global memory)
    CUtensorMap const* tma_Q;       // Q[B,S,NH,D] load
    CUtensorMap const* tma_X;       // X[B,S,H] load
    CUtensorMap const* tma_Wk;      // Wk[NH,D,H] load
    CUtensorMap const* tma_Wv;      // Wv[NH,D,H] load
    CUtensorMap const* tma_Kc;      // Kc[B,S,NH,D] store
    CUtensorMap const* tma_Vc;      // Vc[B,S,NH,D] store
    CUtensorMap const* tma_Kpast;   // Kpast[B,S_past,NH,D] load (or nullptr)
    CUtensorMap const* tma_Vpast;   // Vpast[B,S_past,NH,D] load (or nullptr)
    CUtensorMap const* tma_Opart;   // Opart[B,S,NH,N,D] store
};

////////////////////////////////////////////////////////////////////////////////
// Pipeline barriers
////////////////////////////////////////////////////////////////////////////////
struct alignas(8) PipeBarriers {
    // Projection pipeline (2 stages for X/Wk/Wv)
    uint64_t proj_full[2];   // producer→consumer: data ready
    uint64_t proj_empty[2];  // consumer→producer: buffer free
    // Attention Q pipeline (2 stages)
    uint64_t q_full[2];
    uint64_t q_empty[2];
};

struct PipeState {
    int idx;   // stage index (0 or 1)
    int phase; // phase bit
    __device__ void advance() { if (idx == 1) phase ^= 1; idx ^= 1; }
};

////////////////////////////////////////////////////////////////////////////////
// TMA helpers
////////////////////////////////////////////////////////////////////////////////

// TMA load with mbarrier: cp.async.bulk.tensor.Xd.shared::cluster.global.mbarrier
__device__ __forceinline__ void tma_load_4d(
    uint32_t smem_addr, CUtensorMap const* desc,
    int c0, int c1, int c2, int c3, uint32_t mbar_addr)
{
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
        :: "r"(smem_addr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar_addr));
}

__device__ __forceinline__ void tma_load_3d(
    uint32_t smem_addr, CUtensorMap const* desc,
    int c0, int c1, int c2, uint32_t mbar_addr)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];\n"
        :: "r"(smem_addr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_addr));
}

// TMA store: cp.async.bulk.tensor.Xd.global.shared::cta.bulk_group
__device__ __forceinline__ void tma_store_4d(
    CUtensorMap const* desc, uint32_t smem_addr,
    int c0, int c1, int c2, int c3)
{
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];\n"
        :: "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_addr));
}

__device__ __forceinline__ void tma_store_5d(
    CUtensorMap const* desc, uint32_t smem_addr,
    int c0, int c1, int c2, int c3, int c4)
{
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group"
        " [%0, {%1, %2, %3, %4, %5}], [%6];\n"
        :: "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(smem_addr));
}

__device__ __forceinline__ void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::);
}
__device__ __forceinline__ void tma_store_arrive() {
    asm volatile("cp.async.bulk.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N));
}

// mbarrier helpers
__device__ __forceinline__ void mbar_init(uint64_t* mbar, int count) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* mbar) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(a));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* mbar, int bytes) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, int phase) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "MWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra MWAIT_%=;\n"
        "}\n" :: "r"(a), "r"(phase));
}

////////////////////////////////////////////////////////////////////////////////
// Main warp-specialized kernel
////////////////////////////////////////////////////////////////////////////////
template <typename Traits>
__global__ __launch_bounds__(256, 1)
void proj_fused_fwd_kernel_warpspec(WarpSpecParams params)
{
    using Element = typename Traits::Element;
    constexpr int kBlockM = Traits::kBlockM;       // 64
    constexpr int kBlockN = Traits::kBlockN;       // 64
    constexpr int kHeadDim = Traits::kHeadDim;     // 128
    constexpr int kHidden = Traits::kHiddenDim;    // 128
    constexpr int kHC = Traits::kHiddenChunk;      // 64
    constexpr int kNumChunks = Traits::kNumProjChunks;  // 2
    constexpr int kNThr = Traits::kNThreadsMMA;    // 128
    using SmemLayoutQ = typename Traits::SmemLayoutQ;
    using SmemLayoutK = typename Traits::SmemLayoutK;
    using SmemLayoutV = typename Traits::SmemLayoutV;
    using SmemLayoutX = typename Traits::SmemLayoutX;
    using SmemLayoutW = typename Traits::SmemLayoutW;
    using SmemLayoutO = typename Traits::SmemLayoutO;
    using TiledMmaProj = typename Traits::TiledMmaProj;
    using TiledMmaQK = typename Traits::TiledMmaQK;
    using TiledMmaPV = typename Traits::TiledMmaPV;

    const int n_block = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const int tid = threadIdx.x;
    const int wg = tid / 128;           // 0=producer, 1=consumer
    const int ctid = tid - 128;         // consumer thread id
    const int warp_in_wg = (tid % 128) / 32;

    const int S = params.seqlen, D = params.head_dim, NH = params.num_heads;
    const int H = params.hidden_dim, S_past = params.seqlen_past;
    const int n_start = n_block * kBlockN;
    const int nb_past = params.num_n_blocks_past;
    const bool is_past = (n_block < nb_past);
    const int n_start_new = n_start - S_past;

    float* LSE_ptr = params.ptr_LSE;

    // ---- Shared memory ----
    // Layout: sQ0 | sQ1 | sK | sV | sO0 | sO1 | sX[2] | sWk[2] | sWv[2] | barriers
    // sO0 aliases sWk[0], sO1 aliases sWv[0] during attention (not during projection)
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto a128 = [](uintptr_t p) -> uintptr_t { return (p + 127) & ~uintptr_t(127); };
    uintptr_t b = a128(reinterpret_cast<uintptr_t>(smem_raw));

    constexpr int szQ = cute::cosize(SmemLayoutQ{}) * sizeof(Element);
    constexpr int szK = cute::cosize(SmemLayoutK{}) * sizeof(Element);
    constexpr int szV = cute::cosize(SmemLayoutV{}) * sizeof(Element);
    constexpr int szO = cute::cosize(SmemLayoutO{}) * sizeof(Element);
    constexpr int szX = cute::cosize(SmemLayoutX{}) * sizeof(Element);
    constexpr int szW = cute::cosize(SmemLayoutW{}) * sizeof(Element);

    Element* sQ0_p = (Element*)b;           b = a128(b + szQ);
    Element* sQ1_p = (Element*)b;           b = a128(b + szQ);
    Element* sK_p  = (Element*)b;           b = a128(b + szK);
    Element* sV_p  = (Element*)b;           b = a128(b + szV);
    Element* sO0_p = (Element*)b;           b = a128(b + szO);
    Element* sO1_p = (Element*)b;           b = a128(b + szO);
    // Projection double-buffer: sX[2], sWk[2], sWv[2]
    // Reuse smem: after projection, sWk/sWv are never used again.
    // Stage 0: sX=sQ0, sWk=sO0, sWv=sO1
    // Stage 1: sX=sQ1, sWk=sK,  sWv=sV  (sK/sV only written AFTER projection)
    Element* sX_p[2]  = { sQ0_p, sQ1_p };
    Element* sWk_p[2] = { sO0_p, sK_p  };
    Element* sWv_p[2] = { sO1_p, sV_p  };

    PipeBarriers* pipe = (PipeBarriers*)a128(b);

    auto mk = [](Element* p, auto L) { return make_tensor(make_smem_ptr(p), L); };
    Tensor sQ0 = mk(sQ0_p, SmemLayoutQ{}), sQ1 = mk(sQ1_p, SmemLayoutQ{});
    Tensor sK = mk(sK_p, SmemLayoutK{});
    Tensor sV = mk(sV_p, SmemLayoutV{});
    Tensor sO0 = mk(sO0_p, SmemLayoutO{}), sO1 = mk(sO1_p, SmemLayoutO{});

    constexpr int kTmaQHalf = kBlockM * 64 * sizeof(Element); // Q loaded in 2 halves (d=0..63, d=64..127)
    constexpr int kTmaQTotal = 2 * kTmaQHalf;
    constexpr int kTmaXBytes = kBlockN * kHC * sizeof(Element);
    constexpr int kTmaWBytes = kHeadDim * kHC * sizeof(Element);
    // For K/V TMA: each half = kBlockN * 64 * sizeof(Element)
    constexpr int kTmaKVHalf = kBlockN * 64 * sizeof(Element);
    constexpr int kTmaKVTotal = 2 * kTmaKVHalf;
    // Projection total TMA bytes per stage: X + Wk + Wv
    constexpr int kTmaProjTotal = kTmaXBytes + kTmaWBytes + kTmaWBytes;

    // ---- Init barriers ----
    if (tid == 0) {
        for (int s = 0; s < 2; ++s) {
            mbar_init(&pipe->proj_full[s], 1);
            mbar_init(&pipe->proj_empty[s], 1);
            mbar_arrive(&pipe->proj_empty[s]);  // start empty
            mbar_init(&pipe->q_full[s], 1);
            mbar_init(&pipe->q_empty[s], 1);
            mbar_arrive(&pipe->q_empty[s]);     // start empty
        }
    }
    __syncthreads();

    auto bar128 = [](){ asm volatile("bar.sync 1, 128;\n" ::); };
    const int new_n_idx = n_block - nb_past;
    const int m_start_block = (params.is_causal && new_n_idx > 0) ? new_n_idx : 0;

    // ==================================================================
    // PHASE 1: Projection (TMA pipeline) or Past-KV fetch (TMA load)
    // sX[0] aliases sQ0, sX[1] aliases sQ1 — must complete before Phase 2.
    // ==================================================================
    if (is_past) {
        // Past KV: TMA load K directly into sK, V into sQ0 staging then transpose.
        // K layout in global: [B,S_past,NH,D] row-major → matches sK [kBlockN,kHeadDim] K-major.
        // V layout in global: same row-major → need transpose to sV [kHeadDim,kBlockN] MN-major.
        // Use sQ0 as temporary staging buffer for V (same swizzle as sK).
        if (tid == 0) {
            uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(&pipe->proj_full[0]));
            mbar_arrive_expect_tx(&pipe->proj_full[0], 2 * kTmaKVTotal);
            // K: two half-loads (d=0..63, d=64..127) directly into sK
            uint32_t sk = static_cast<uint32_t>(__cvta_generic_to_shared(sK_p));
            tma_load_4d(sk,              params.tma_Kpast, 0,  head, n_start, batch, fb);
            tma_load_4d(sk + kTmaKVHalf, params.tma_Kpast, 64, head, n_start, batch, fb);
            // V: two half-loads into sQ0 staging buffer (row-major, will transpose)
            uint32_t sv_stg = static_cast<uint32_t>(__cvta_generic_to_shared(sQ0_p));
            tma_load_4d(sv_stg,              params.tma_Vpast, 0,  head, n_start, batch, fb);
            tma_load_4d(sv_stg + kTmaKVHalf, params.tma_Vpast, 64, head, n_start, batch, fb);
        }
        // All threads wait for TMA completion
        mbar_wait(&pipe->proj_full[0], 0);
        // Transpose V from sQ0 staging [kBlockN,kHeadDim] K-major → sV [kHeadDim,kBlockN] MN-major
        {
            Tensor sVstg = mk(sQ0_p, SmemLayoutK{});
            constexpr int kEPV = 8, kVPR = kHeadDim / kEPV;
            constexpr int kTV = kBlockN * kVPR, kVPT = (kTV + 255) / 256;
            #pragma unroll
            for (int v = 0; v < kVPT; ++v) {
                int idx = v * 256 + tid;
                if (idx < kTV) {
                    int i = idx / kVPR, d = (idx % kVPR) * kEPV;
                    #pragma unroll
                    for (int k = 0; k < kEPV; ++k)
                        sV(d + k, i) = sVstg(i, d + k);
                }
            }
        }
        __syncthreads();
    } else {
        // Projection with TMA pipeline: WG0 warp 0 = TMA producer, WG1 = WGMMA consumer.
        // Double-buffered: sX_p[0/1], sWk_p[0/1], sWv_p[0/1] with proj_full/proj_empty barriers.
        // kNumChunks iterations over hidden-dim chunks (e.g., 2 for H=128, 4 for H=256).
        constexpr int kStages = 2;

        if (wg == 0) {
            // === WG0: TMA PRODUCER (warp 0, lane 0 only) ===
            if (warp_in_wg == 0 && tid == 0) {
                PipeState pp = {0, 0};
                // Pre-load chunk 0 → stage 0
                {
                    uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(&pipe->proj_full[0]));
                    mbar_arrive_expect_tx(&pipe->proj_full[0], kTmaProjTotal);
                    tma_load_3d(static_cast<uint32_t>(__cvta_generic_to_shared(sX_p[0])),
                                params.tma_X,  0, n_start_new, batch, fb);
                    tma_load_3d(static_cast<uint32_t>(__cvta_generic_to_shared(sWk_p[0])),
                                params.tma_Wk, 0, 0,           head,  fb);
                    tma_load_3d(static_cast<uint32_t>(__cvta_generic_to_shared(sWv_p[0])),
                                params.tma_Wv, 0, 0,           head,  fb);
                }
                pp.advance();
                // Pipeline: pre-load chunk cs+1 while consumer processes chunk cs
                for (int cs = 0; cs < kNumChunks - 1; ++cs) {
                    int nxt = pp.idx;
                    // Wait for consumer to free this stage (skip for first kStages loads)
                    if (cs + 1 >= kStages)
                        mbar_wait(&pipe->proj_empty[nxt], pp.phase);
                    uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(&pipe->proj_full[nxt]));
                    mbar_arrive_expect_tx(&pipe->proj_full[nxt], kTmaProjTotal);
                    int h_off = (cs + 1) * kHC;
                    tma_load_3d(static_cast<uint32_t>(__cvta_generic_to_shared(sX_p[nxt])),
                                params.tma_X,  h_off, n_start_new, batch, fb);
                    tma_load_3d(static_cast<uint32_t>(__cvta_generic_to_shared(sWk_p[nxt])),
                                params.tma_Wk, h_off, 0,           head,  fb);
                    tma_load_3d(static_cast<uint32_t>(__cvta_generic_to_shared(sWv_p[nxt])),
                                params.tma_Wv, h_off, 0,           head,  fb);
                    pp.advance();
                }
            }
            // WG0 other threads: idle during projection
        } else {
            // === WG1: WGMMA CONSUMER ===
            TiledMmaProj mma_proj;
            auto thr_proj = mma_proj.get_thread_slice(ctid);
            Tensor acc_k = partition_fragment_C(mma_proj, Shape<Int<kBlockN>,Int<kHeadDim>>{});
            Tensor acc_v = partition_fragment_C(mma_proj, Shape<Int<kBlockN>,Int<kHeadDim>>{});
            clear(acc_k); clear(acc_v);

            PipeState pc = {0, 0};
            #pragma unroll
            for (int cs = 0; cs < kNumChunks; ++cs) {
                int stg = pc.idx;
                // Wait for TMA producer to fill this stage
                mbar_wait(&pipe->proj_full[stg], pc.phase);
                // Create fragments from the current stage's smem buffers
                Tensor sX_cur  = mk(sX_p[stg],  SmemLayoutX{});
                Tensor sWk_cur = mk(sWk_p[stg], SmemLayoutW{});
                Tensor sWv_cur = mk(sWv_p[stg], SmemLayoutW{});
                auto tA_cur  = thr_proj.partition_fragment_A(sX_cur);
                auto tBk_cur = thr_proj.partition_fragment_B(sWk_cur);
                auto tBv_cur = thr_proj.partition_fragment_B(sWv_cur);
                if (cs == 0) {
                    flash::gemm<true,-1>(mma_proj, tA_cur, tBk_cur, acc_k);
                    flash::gemm<true,-1>(mma_proj, tA_cur, tBv_cur, acc_v);
                } else {
                    flash::gemm<false,-1>(mma_proj, tA_cur, tBk_cur, acc_k);
                    flash::gemm<false,-1>(mma_proj, tA_cur, tBv_cur, acc_v);
                }
                cute::warpgroup_wait<0>();
                cute::warpgroup_fence_operand(acc_k);
                cute::warpgroup_fence_operand(acc_v);
                // Signal producer that this stage is free for reuse
                if (ctid == 0) mbar_arrive(&pipe->proj_empty[stg]);
                pc.advance();
            }
            // Write accumulator results to sK and sV
            auto tCsK = thr_proj.partition_C(sK);
            Tensor cV = make_identity_tensor(Shape<Int<kBlockN>,Int<kHeadDim>>{});
            auto tCcV = thr_proj.partition_C(cV);
            #pragma unroll
            for (int i = 0; i < size(acc_k); ++i) {
                tCsK(i) = Element(acc_k(i));
                sV(get<1>(tCcV(i)), get<0>(tCcV(i))) = Element(acc_v(i));
            }
        }
    }
    __syncthreads(); // sQ0/sQ1 now free — producer can TMA into them

    // KV cache TMA store: K directly from sK, V transposed via sQ0 staging
    if (!is_past) {
        // Transpose sV [kHeadDim,kBlockN] MN-major → sQ0 [kBlockN,kHeadDim] K-major
        {
            Tensor sVstg = mk(sQ0_p, SmemLayoutK{});
            constexpr int kEPV = 8, kVPR = kHeadDim / kEPV;
            constexpr int kTV = kBlockN * kVPR, kVPT = (kTV + 255) / 256;
            #pragma unroll
            for (int v = 0; v < kVPT; ++v) {
                int idx = v * 256 + tid;
                if (idx < kTV) {
                    int i = idx / kVPR, d = (idx % kVPR) * kEPV;
                    #pragma unroll
                    for (int k = 0; k < kEPV; ++k)
                        sVstg(i, d + k) = sV(d + k, i);
                }
            }
        }
        __syncthreads(); // transpose complete before TMA reads smem
        if (tid == 0) {
            tma_store_fence();
            // K: two halves from sK
            uint32_t sk = static_cast<uint32_t>(__cvta_generic_to_shared(sK_p));
            tma_store_4d(params.tma_Kc, sk,              0,  head, n_start_new, batch);
            tma_store_4d(params.tma_Kc, sk + kTmaKVHalf, 64, head, n_start_new, batch);
            // V: two halves from sQ0 staging (now row-major, matches tma_Vc descriptor)
            uint32_t sv = static_cast<uint32_t>(__cvta_generic_to_shared(sQ0_p));
            tma_store_4d(params.tma_Vc, sv,              0,  head, n_start_new, batch);
            tma_store_4d(params.tma_Vc, sv + kTmaKVHalf, 64, head, n_start_new, batch);
            tma_store_arrive();
            tma_store_wait<0>(); // wait for store completion before smem reuse in Phase 2
        }
    }
    __syncthreads(); // Ensure all stores complete before warpgroup divergence

    // ==================================================================
    // PHASE 2: Warpgroup divergence for attention
    // ==================================================================
    if (wg == 0) {
        // ============ PRODUCER: TMA Q loader ============
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n" ::);
        if (warp_in_wg != 0) return;
        bool is_t0 = (tid == 0);
        PipeState pw = {0, 0};
        for (int mb = m_start_block; mb < params.num_m_blocks; ++mb) {
            int stg = pw.idx;
            if (mb >= m_start_block + 2)
                mbar_wait(&pipe->q_empty[stg], pw.phase);
            if (is_t0) {
                mbar_arrive_expect_tx(&pipe->q_full[stg], kTmaQTotal);
                uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(&pipe->q_full[stg]));
                Element* sq = (stg == 0) ? sQ0_p : sQ1_p;
                uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(sq));
                tma_load_4d(s0, params.tma_Q, 0, head, mb * kBlockM, batch, fb);
                tma_load_4d(s0 + kTmaQHalf, params.tma_Q, 64, head, mb * kBlockM, batch, fb);
            }
            pw.advance();
        }
        return;
    }
    // ============ CONSUMER: Attention ============
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n" ::);
    float const sscale = params.softmax_scale * float(M_LOG2E);
    constexpr int kNR = 2 * (kBlockM / 64);
    using SoftT = flash::Softmax<kNR>;

    TiledMmaQK mma_qk;
    TiledMmaPV mma_pv;
    auto thr_qk = mma_qk.get_thread_slice(ctid);
    auto thr_pv = mma_pv.get_thread_slice(ctid);
    Tensor tSrQ0 = thr_qk.partition_fragment_A(sQ0);
    Tensor tSrQ1 = thr_qk.partition_fragment_A(sQ1);
    Tensor tSrK  = thr_qk.partition_fragment_B(sK);
    Tensor tOrV  = thr_pv.partition_fragment_B(sV);

    auto smem_copy_O = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, mma_pv);
    auto sthr_O = smem_copy_O.get_thread_slice(ctid);
    Tensor tOsO0 = sthr_O.partition_D(sO0);
    Tensor tOsO1 = sthr_O.partition_D(sO1);

    const int nbo = params.num_n_blocks;
    constexpr int kTmaOHalf = kBlockM * 64 * sizeof(Element); // O half-tile bytes

    // Write LSE=-INF for skipped m_blocks
    for (int mb = 0; mb < m_start_block; ++mb) {
        int ms = mb * kBlockM;
        for (int i = ctid; i < kBlockM; i += kNThr)
            LSE_ptr[(((size_t)batch * S + ms + i) * NH + head) * nbo + n_block] = -INFINITY;
    }

    // Consumer waits on producer's TMA pipeline
    PipeState qr = {0, 0};

    for (int mb = m_start_block; mb < params.num_m_blocks; ++mb) {
        int ms = mb * kBlockM;
        int buf = qr.idx;
        int obuf = mb & 1;

        // Wait for Q tile from producer
        mbar_wait(&pipe->q_full[buf], qr.phase);
        bar128();

        // QK GEMM (async, wg_wait=-1)
        Tensor acc_s = partition_fragment_C(mma_qk, Shape<Int<kBlockM>,Int<kBlockN>>{});
        if (buf==0) flash::gemm<true,-1>(mma_qk, tSrQ0, tSrK, acc_s);
        else        flash::gemm<true,-1>(mma_qk, tSrQ1, tSrK, acc_s);

        // Overlap: TMA store previous O to global while QK runs
        if (mb > m_start_block && ctid == 0) {
            // Wait for any outstanding TMA store on the buffer we're about to reuse
            // (from 2 iterations ago). tma_store_wait<1>.read guarantees smem reads done.
            if (mb > m_start_block + 1)
                tma_store_wait<1>();
            int prev_obuf = (mb-1) & 1;
            int pms = (mb-1) * kBlockM;
            Element* sOp = (prev_obuf==0) ? sO0_p : sO1_p;
            uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(sOp));
            tma_store_fence();
            tma_store_5d(params.tma_Opart, sa,              0,  n_block, head, pms, batch);
            tma_store_5d(params.tma_Opart, sa + kTmaOHalf,  64, n_block, head, pms, batch);
            tma_store_arrive();
        }

        // Wait for QK GEMM
        cute::warpgroup_wait<0>(); cute::warpgroup_fence_operand(acc_s);

        // Release Q buffer: signal producer that this buffer is free
        bar128();
        if (ctid == 0) mbar_arrive(&pipe->q_empty[buf]);
        qr.advance();

        // Causal mask
        if (params.is_causal) {
            auto t0 = TiledMmaQK{}.get_thread_slice(_0{});
            Tensor cS = make_identity_tensor(Shape<Int<kBlockM>,Int<kBlockN>>{});
            Tensor tS = thr_qk.partition_C(cS);
            Tensor t0S = t0.partition_C(cS);
            Tensor ar = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Tensor tr = make_tensor(tS.data(), flash::convert_layout_acc_rowcol(tS.layout()));
            Tensor t0r = make_tensor(t0S.data(), flash::convert_layout_acc_rowcol(t0S.layout()));
            int tc = get<1>(tr(_0{},_0{}));
            int cr = S_past + 1 - n_start + ms - tc;
            #pragma unroll
            for (int m = 0; m < size<0>(ar); ++m) {
                int cl = get<0>(tr(m,_0{})) + cr;
                #pragma unroll
                for (int n = 0; n < size<1>(ar); ++n)
                    if (get<1>(t0r(_0{},n)) >= cl) ar(m,n) = -INFINITY;
            }
        }

        // Softmax (Is_first=true: each Q block is independent)
        SoftT softmax(sscale);
        (void)softmax.template max_get_scale<true, true>(acc_s);
        softmax.template online_softmax<true, true>(acc_s);

        // Convert to P fp16
        auto tP = make_tensor(acc_s.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(acc_s.layout()));
        Tensor tOrP = make_tensor_like<Element>(tP);
        flash::convert_type_out(tP, tOrP);

        // PV GEMM (fresh acc_o for each Q block)
        Tensor acc_o = partition_fragment_C(mma_pv, Shape<Int<kBlockM>,Int<kHeadDim>>{});
        flash::gemm<true,-1>(mma_pv, tOrP, tOrV, acc_o);

        // Finalize softmax + LSE store
        auto fscale = softmax.finalize();
        {
            Tensor cOl = make_identity_tensor(Shape<Int<kBlockM>,Int<kHeadDim>>{});
            auto tOl = thr_pv.partition_C(cOl);
            if (ctid % 32 % 4 == 0) {
                auto tOrc = make_tensor(tOl.data(), flash::convert_layout_acc_rowcol(tOl.layout()));
                #pragma unroll
                for (int mi = 0; mi < kNR; ++mi) {
                    int row = ms + get<0>(tOrc(mi,_0{}));
                    LSE_ptr[(((size_t)batch*S+row)*NH+head)*nbo+n_block] = softmax.row_sum(mi);
                }
            }
        }

        // Wait for PV, rescale O
        cute::warpgroup_wait<0>(); cute::warpgroup_fence_operand(acc_o);
        softmax.rescale_o(acc_o, fscale);

        // STSM O to smem (double-buffered)
        {
            Tensor rO = make_tensor_like<Element>(acc_o);
            flash::convert_type_out(acc_o, rO);
            auto tR = sthr_O.retile_S(rO);
            cute::copy(smem_copy_O, tR, (obuf==0) ? tOsO0 : tOsO1);
            bar128();
        }
    }

    // TMA store last O block
    if (m_start_block < params.num_m_blocks) {
        if (ctid == 0) {
            tma_store_wait<0>(); // wait for all prior O stores to finish
            int last_obuf = (params.num_m_blocks-1) & 1;
            int lms = (params.num_m_blocks-1) * kBlockM;
            Element* sOl = (last_obuf==0) ? sO0_p : sO1_p;
            uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(sOl));
            tma_store_fence();
            tma_store_5d(params.tma_Opart, sa,              0,  n_block, head, lms, batch);
            tma_store_5d(params.tma_Opart, sa + kTmaOHalf,  64, n_block, head, lms, batch);
            tma_store_arrive();
            tma_store_wait<0>(); // wait for last store to complete before kernel exit
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// TMA descriptor creation helpers (host-side)
////////////////////////////////////////////////////////////////////////////////

// Q: [B, S, NH, D] → TMA 4D: (D, NH, S, B), box={64,1,kBlockM,1}
inline CUresult create_tma_Q(CUtensorMap* desc, void const* ptr,
    int B, int S, int NH, int D) {
    uint64_t gd[4]={(uint64_t)D,(uint64_t)NH,(uint64_t)S,(uint64_t)B};
    uint64_t gs[3]={(uint64_t)D*2,(uint64_t)NH*D*2,(uint64_t)S*NH*D*2};
    uint32_t bx[4]={64,1,64,1}; uint32_t es[4]={1,1,1,1};
    return cuTensorMapEncodeTiled(desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,4,
        const_cast<void*>(ptr),gd,gs,bx,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// X: [B, S, H] → TMA 3D: (H, S, B), box={kHC, kBlockN, 1}
inline CUresult create_tma_X(CUtensorMap* desc, void const* ptr,
    int B, int S, int H, int kHC, int kBlockN) {
    uint64_t gd[3]={(uint64_t)H,(uint64_t)S,(uint64_t)B};
    uint64_t gs[2]={(uint64_t)H*2,(uint64_t)S*H*2};
    uint32_t bx[3]={(uint32_t)kHC,(uint32_t)kBlockN,1}; uint32_t es[3]={1,1,1};
    return cuTensorMapEncodeTiled(desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,3,
        const_cast<void*>(ptr),gd,gs,bx,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// W: [NH, D, H] → TMA 3D: (H, D, NH), box={kHC, kHeadDim, 1}
inline CUresult create_tma_W(CUtensorMap* desc, void const* ptr,
    int NH, int D, int H, int kHC) {
    uint64_t gd[3]={(uint64_t)H,(uint64_t)D,(uint64_t)NH};
    uint64_t gs[2]={(uint64_t)H*2,(uint64_t)D*H*2};
    uint32_t bx[3]={(uint32_t)kHC,(uint32_t)D,1}; uint32_t es[3]={1,1,1};
    return cuTensorMapEncodeTiled(desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,3,
        const_cast<void*>(ptr),gd,gs,bx,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// KV: [B, S, NH, D] → TMA 4D: (D, NH, S, B), box={64,1,kBlockN,1}
inline CUresult create_tma_KV(CUtensorMap* desc, void const* ptr,
    int B, int S, int NH, int D, int kBlockN) {
    uint64_t gd[4]={(uint64_t)D,(uint64_t)NH,(uint64_t)S,(uint64_t)B};
    uint64_t gs[3]={(uint64_t)D*2,(uint64_t)NH*D*2,(uint64_t)S*NH*D*2};
    uint32_t bx[4]={64,1,(uint32_t)kBlockN,1}; uint32_t es[4]={1,1,1,1};
    return cuTensorMapEncodeTiled(desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,4,
        const_cast<void*>(ptr),gd,gs,bx,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// Opart: [B, S, NH, N, D] → TMA 5D: (D, N, NH, S, B), box={64,1,1,kBlockM,1}
inline CUresult create_tma_Opart(CUtensorMap* desc, void* ptr,
    int B, int S, int NH, int N, int D, int kBlockM) {
    uint64_t gd[5]={(uint64_t)D,(uint64_t)N,(uint64_t)NH,(uint64_t)S,(uint64_t)B};
    uint64_t gs[4]={(uint64_t)D*2,(uint64_t)N*D*2,(uint64_t)NH*N*D*2,(uint64_t)S*NH*N*D*2};
    uint32_t bx[5]={64,1,1,(uint32_t)kBlockM,1}; uint32_t es[5]={1,1,1,1,1};
    return cuTensorMapEncodeTiled(desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,5,
        ptr,gd,gs,bx,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// ---------------------------------------------------------------------------
// bf16 TMA descriptor helpers — same shape/strides as fp16 (16-bit element),
// only the CU_TENSOR_MAP_DATA_TYPE differs.
// ---------------------------------------------------------------------------
inline CUresult create_tma_Q_bf16(CUtensorMap* desc, void const* ptr,
    int B, int S, int NH, int D) {
    uint64_t gd[4]={(uint64_t)D,(uint64_t)NH,(uint64_t)S,(uint64_t)B};
    uint64_t gs[3]={(uint64_t)D*2,(uint64_t)NH*D*2,(uint64_t)S*NH*D*2};
    uint32_t bx[4]={64,1,64,1}; uint32_t es[4]={1,1,1,1};
    return cuTensorMapEncodeTiled(desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,4,
        const_cast<void*>(ptr),gd,gs,bx,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline CUresult create_tma_X_bf16(CUtensorMap* desc, void const* ptr,
    int B, int S, int H, int kHC, int kBlockN) {
    uint64_t gd[3]={(uint64_t)H,(uint64_t)S,(uint64_t)B};
    uint64_t gs[2]={(uint64_t)H*2,(uint64_t)S*H*2};
    uint32_t bx[3]={(uint32_t)kHC,(uint32_t)kBlockN,1}; uint32_t es[3]={1,1,1};
    return cuTensorMapEncodeTiled(desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,
        const_cast<void*>(ptr),gd,gs,bx,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline CUresult create_tma_W_bf16(CUtensorMap* desc, void const* ptr,
    int NH, int D, int H, int kHC) {
    uint64_t gd[3]={(uint64_t)H,(uint64_t)D,(uint64_t)NH};
    uint64_t gs[2]={(uint64_t)H*2,(uint64_t)D*H*2};
    uint32_t bx[3]={(uint32_t)kHC,(uint32_t)D,1}; uint32_t es[3]={1,1,1};
    return cuTensorMapEncodeTiled(desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,
        const_cast<void*>(ptr),gd,gs,bx,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

} // namespace proj_fused
