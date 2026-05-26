/******************************************************************************
 * SM-Parallel v2 — CPU KV Cache variant (SM90, bf16) — GH200 NVLink-C2C build
 *
 * KV cache lives in pinned CPU memory (cudaMallocHost / NVLink-C2C NUMA memory).
 * Past tokens:  single-buffer TMA load CPU→smem.  NVLink-C2C latency ≈ 0.24 µs
 *               per 32 KB tile; Q-inner compute ≈ 4–16 µs → no double-buffer needed.
 * New tokens:   GPU project X→K,V; async TMA store smem→CPU (overlapped w/ Q-inner).
 * V layout in CPU: [B, S_total, NH, kD] token-major (D fastest) — same as K.
 *               Box = {64, 1, 8, 1} per smem atom.  128B swizzle requires box[0]=64 (128B/row).
 *
 * Loop order: KV-outer / Q-inner (preserved from base SMP v2).
 * Smem:  single sK_buf + sV_buf = ~160 KB total (fits 228 KB; was 192 KB with
 *         double-buffer), allowing 2 CTAs/SM for improved occupancy.
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

namespace sm_parallel_v2_cpu_kv {

using namespace cute;

// ---------------------------------------------------------------------------
// TMA helpers
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

// TMA bulk store: smem → global (CPU pinned or HBM).
// Coordinates: (c0..c3) in the descriptor's dimension ordering.
__device__ __forceinline__ void tma_store_4d(
    CUtensorMap const* desc, int c0, int c1, int c2, int c3, uint32_t smem_addr)
{
    asm volatile(
        "cp.async.bulk.tensor.4d.bulk_group.global.shared::cta"
        " [%0, {%1, %2, %3, %4}], [%5];\n"
        :: "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_addr));
}

// Required on SM90 before cp.async.bulk stores to make prior WGMMA smem
// writes visible to the TMA store engine.
__device__ __forceinline__ void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

// Wait until at most N bulk-copy groups remain outstanding (N must be a
// compile-time constant — PTX immediate).
template <int N>
__device__ __forceinline__ void cp_async_bulk_wait_group() {
    asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void mbar_init(uint64_t* mbar, int count) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(count));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* mbar, int bytes) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(a), "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, int phase) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    // .shared::cta scope provides proxy-acquire fence for cp.async.bulk (SM90).
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

enum class SmpV2Barrier : uint32_t { OSmemReady = 0, MergeStatsReady = 1, QRotReady = 2 };

// ---------------------------------------------------------------------------
// Params struct
// ---------------------------------------------------------------------------
struct SmParallelV2CpuKVParams {
    // GPU HBM inputs
    void const* ptr_X;        // [B, S_new, kHiddenDim] bf16
    void const* ptr_Wk;       // [NH, kHeadDim, kHiddenDim] bf16
    void const* ptr_Wv;       // [NH, kHeadDim, kHiddenDim] bf16
    void const* ptr_Q;        // [B, S_q, NH, kHeadDim] bf16
    // GPU HBM outputs
    void*       ptr_O;        // [B, S_q, NH, kHeadDim] bf16
    float*      ptr_O_run;    // [B, S_q, NH, kHeadDim] fp32, init=0
    float*      ptr_LSE_run;  // [B, S_q, NH] fp32, init=-INF

    // CPU pinned KV cache (cudaMallocHost / NVLink-C2C NUMA memory)
    //   K: [B, S_total, NH, kD]  token-major bf16  (D fastest)
    //   V: [B, S_total, NH, kD]  token-major bf16  (D fastest, same as K)
    // For new tokens: kernel writes projected K,V at the new-token slice.
    // For past tokens: kernel reads K,V from [*, 0..seqlen_past, *, *].
    void const* ptr_K_cpu;
    void*       ptr_V_cpu;

    int batch, seqlen_new, seqlen_q, num_heads, head_dim, hidden_dim;
    int seqlen_past;           // 0 for pure prefill
    int num_m_blocks, num_n_blocks, num_n_blocks_past;
    float softmax_scale;
    int is_causal;

    // GQA: num_q_heads = NH_q, gqa_ratio = NH_q / num_heads (1 for MHA)
    // Grid is (num_heads=NH_kv, batch); each CTA handles gqa_ratio Q-heads.
    int num_q_heads;   // = NH_q
    int gqa_ratio;     // = NH_q / num_heads  (1 for MHA)

    // CUDA-graph support: if non-null, kernel reads num_n_blocks_past and
    // num_n_blocks from device memory (updated by host before graph replay).
    // Both pointers must live in GPU-accessible memory.
    int const* ptr_num_n_blocks_past;   // device int32, or nullptr
    int const* ptr_num_n_blocks;        // device int32, or nullptr

    // RoPE fields: ptr_cos_sin = null disables rotation (OPT models).
    // cos_sin_cache layout: [max_position, head_dim] fp32, row p =
    //   [cos(p*θ_0)..cos(p*θ_{D/2-1}), sin(p*θ_0)..sin(p*θ_{D/2-1})]
    float const* ptr_cos_sin;   // GPU HBM; null = no RoPE
    int   rope_mode;            // 0=none, 1=neox_style
    int   rope_dim;             // rotary dims (= head_dim for full RoPE)
    int   q_pos_start;          // global position index of the first Q token

    // TMA descriptors (on device, created by host launcher)
    CUtensorMap const* tma_X;
    CUtensorMap const* tma_Wk;
    CUtensorMap const* tma_Wv;
    CUtensorMap const* tma_Q;
    CUtensorMap const* tma_K_cpu;   // K: globalDim={kD,NH,S_total,n_pool}  box={64,1,kBN,1}
    CUtensorMap const* tma_V_cpu;   // V: globalDim={kD,NH,S_total,n_pool}  box={64,1,8,1} (atom)

    // Pool-row remapping for CUDA graph decode.
    // ptr_req_pool_indices[b] = pool row for batch position b (device int32[batch]).
    // Null in the non-graph path — kernel uses batch (= blockIdx.y) directly.
    int const* ptr_req_pool_indices;
};

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
template <typename Traits>
__global__ __launch_bounds__(256, 1)
void sm_parallel_v2_cpu_kv_fwd_kernel(SmParallelV2CpuKVParams params)
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

    using SmemLayoutAtomW_full = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element,
            cute::Int<Traits::kHeadDim>, cute::Int<Traits::kHiddenDim>>());
    using SmemLayoutW_full = decltype(cute::tile_to_shape(
        SmemLayoutAtomW_full{},
        cute::make_shape(cute::Int<Traits::kHeadDim>{},
                         cute::Int<Traits::kHiddenDim>{})));

    const int head      = blockIdx.x;
    const int batch     = blockIdx.y;
    // Pool-row remapping: during CUDA graph decode, pool row ≠ batch position.
    const int pool_batch = (params.ptr_req_pool_indices != nullptr)
        ? params.ptr_req_pool_indices[batch]
        : batch;
    const int tid       = threadIdx.x;
    const int wg_idx    = tid / 128;
    const int ctid      = tid - 128;   // valid when wg_idx == 1
    const int NH        = params.num_heads;      // KV heads (= NH_kv)
    const int NH_q      = params.num_q_heads;    // Q heads (= gqa_ratio * NH)
    const int gqa_ratio = params.gqa_ratio;      // Q-heads per KV-head (1 for MHA)
    const int S_q       = params.seqlen_q;

    auto* O_final_ptr  = reinterpret_cast<Element*>(params.ptr_O);
    float* O_run_ptr   = params.ptr_O_run;
    float* LSE_run_ptr = params.ptr_LSE_run;

    // ------------------------------------------------------------------
    // Shared memory layout (~160 KB, GH200 single-buffer):
    //   sQ0, sQ1         [kBlockM, kHeadDim]  SmemLayoutQ   2×16KB  Q buf / X staging
    //   sWk_full         [kHeadDim, kHidDim]  SmemLayoutW_full 32KB persistent Wk
    //   sWv_full         [kHeadDim, kHidDim]  SmemLayoutW_full 32KB persistent Wv
    //   sKbuf            [kBlockN, kHeadDim]  SmemLayoutK   16KB    KV single-buf
    //   sVbuf            [kHeadDim, kBlockN]  SmemLayoutV   16KB    KV single-buf
    //   sO0, sO1         [kBlockM, kHeadDim]  SmemLayoutO   2×16KB  output staging
    //   startup_mbar     uint64_t              8B
    //   kv_mbar          uint64_t              8B             CPU KV load barrier
    //   pipe_proj_smem   PipelineProj::SS
    //   pipe_q_smem      PipelineQ::SS
    //   sLSE_local       float[kBlockM]        256B
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

    // Single-buffer K and V smem regions (GH200: NVLink-C2C latency ≈ Q-inner/65)
    Element* sKbuf = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutK{}) * sizeof(Element));
    Element* sVbuf = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutV{}) * sizeof(Element));

    Element* sO0_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutO{}) * sizeof(Element));
    Element* sO1_ptr = reinterpret_cast<Element*>(base);
    base = align128(base + cute::cosize(SmemLayoutO{}) * sizeof(Element));

    // Barriers
    base = (base + 7) & ~uintptr_t(7);
    uint64_t* startup_mbar = reinterpret_cast<uint64_t*>(base);
    base += 8;
    base = (base + 7) & ~uintptr_t(7);
    uint64_t* kv_mbar = reinterpret_cast<uint64_t*>(base);   // single KV load barrier
    base += 8;
    base = (base + 15) & ~uintptr_t(15);
    auto* pipe_proj_smem = reinterpret_cast<PipelineProj::SharedStorage*>(base);
    base += sizeof(PipelineProj::SharedStorage);
    base = (base + 15) & ~uintptr_t(15);
    auto* pipe_q_smem = reinterpret_cast<PipelineQ::SharedStorage*>(base);
    base += sizeof(PipelineQ::SharedStorage);
    base = (base + 15) & ~uintptr_t(15);
    float* sLSE_local = reinterpret_cast<float*>(base);

    // CuTe views for attention output staging
    Tensor sO0 = make_tensor(make_smem_ptr(sO0_ptr), SmemLayoutO{});
    Tensor sO1 = make_tensor(make_smem_ptr(sO1_ptr), SmemLayoutO{});

    // sX_p[stage]: X staging buffers alias sQ0/sQ1 during projection subphase
    Element* sX_p[2] = { sQ0_ptr, sQ1_ptr };

    // TMA byte constants
    constexpr int kTmaXBytes   = kBlockN  * kHiddenChunk * (int)sizeof(Element);
    constexpr int kTmaWBytes   = kHeadDim * kHiddenChunk * (int)sizeof(Element);
    constexpr int kTmaQHalf    = kBlockM  * 64            * (int)sizeof(Element);
    // Half of one K tile in smem (64 D-elements × kBlockN tokens)
    constexpr int kTmaKVHalf   = kBlockN  * 64            * (int)sizeof(Element);
    // V atom constants: SmemLayoutV MN-major atom = [kD_atom=64, kBN_atom=8] for bf16 128B swizzle.
    // Outer tiling [kD=128, kBN=64] uses column-major (d_outer fastest), interleaving D-halves.
    // smem offset of atom (d_outer, s_outer) = (d_outer + s_outer * 2) * kVAtomBytes.
    constexpr int kVAtomS     = 8;   // kBN_atom = 8 for bf16 128B swizzle MN-major
    constexpr int kVNSAtoms   = kBlockN / kVAtomS;   // 8 S-atom groups per tile
    constexpr int kVAtomBytes = 64 * kVAtomS * (int)sizeof(Element);  // 1024 bytes per atom
    // Total bytes per CPU KV tile: K (2 D-halves × 8KB each = 16KB) + V (16 atoms × 1KB = 16KB)
    constexpr int kKVTileBytes  = 4 * kTmaKVHalf;  // = 32768 bytes (16KB K + 16KB V)

    constexpr int kChunkStrideW = cute::cosize(SmemLayoutW{});

    // ------------------------------------------------------------------
    // Pipeline construction (all 256 threads)
    // ------------------------------------------------------------------
    PipelineProj::Params proj_p;
    proj_p.role              = (wg_idx == 0) ? PipelineProj::ThreadCategory::Producer
                                              : PipelineProj::ThreadCategory::Consumer;
    proj_p.transaction_bytes = static_cast<uint32_t>(kTmaXBytes);
    proj_p.is_leader         = (wg_idx == 0 && tid == 0) ? 1u : 0u;
    proj_p.num_consumers     = static_cast<uint32_t>(kNThreadsMMA);
    PipelineProj pipeline_proj(*pipe_proj_smem, proj_p, Shape<_1,_1,_1>{});

    PipelineQ::Params q_p;
    q_p.role              = (wg_idx == 0) ? PipelineQ::ThreadCategory::Producer
                                           : PipelineQ::ThreadCategory::Consumer;
    q_p.transaction_bytes = 2u * static_cast<uint32_t>(kTmaQHalf);
    q_p.is_leader         = (wg_idx == 0 && tid == 0) ? 1u : 0u;
    q_p.num_consumers     = static_cast<uint32_t>(kNThreadsMMA);
    PipelineQ pipeline_q(*pipe_q_smem, q_p, Shape<_1,_1,_1>{});

    // Init startup_mbar and kv_mbar (tid==0; __syncthreads ensures visibility)
    if (tid == 0) {
        mbar_init(startup_mbar, 1);
        mbar_init(kv_mbar,      1);
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // STARTUP: load Wk_full and Wv_full for this head into persistent smem
    // ------------------------------------------------------------------
    {
        constexpr int kStartupBytes = kNumProjChunks * kTmaWBytes * 2;

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
        mbar_wait(startup_mbar, 0);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
    }

    // ------------------------------------------------------------------
    // Pipeline states (declared outside loop for phase continuity)
    // ------------------------------------------------------------------
    auto pipe_proj_write = cutlass::make_producer_start_state<PipelineProj>();
    cutlass::PipelineState<2> pipe_proj_read{};
    auto pipe_q_write = cutlass::make_producer_start_state<PipelineQ>();
    cutlass::PipelineState<2> pipe_q_read{};

    const float softmax_scale_log2 = params.softmax_scale * float(M_LOG2E);
    constexpr int kNRows = 2 * (kBlockM / 64);

    // Single-buffer state: one phase counter for kv_mbar
    int kv_phase = 0;

    // Hoist dynamic loop bounds (support CUDA graph via device-memory pointers)
    const int num_n_blocks      = params.ptr_num_n_blocks
                                    ? *params.ptr_num_n_blocks
                                    : params.num_n_blocks;
    const int num_n_blocks_past = params.ptr_num_n_blocks_past
                                    ? *params.ptr_num_n_blocks_past
                                    : params.num_n_blocks_past;

    // ------------------------------------------------------------------
    // Outer loop: KV-outer / Q-inner
    // ------------------------------------------------------------------
    for (int n_block = 0; n_block < num_n_blocks; ++n_block) {
        const int n_start  = n_block * kBlockN;
        const bool is_past = (n_block < num_n_blocks_past);

        // ==============================================================
        // STEP 1: Produce sKbuf[cur] / sVbuf[cur]
        // ==============================================================
        if (is_past) {
            // GH200 NVLink-C2C: issue TMA load for THIS tile, then wait.
            // NVLink-C2C latency ≈ 0.24 µs vs Q-inner ≈ 4–16 µs → negligible stall.
            // Single-buffer: saves 32 KB smem → 2 CTAs/SM occupancy.
            if (tid == 0) {
                uint32_t mbar_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(kv_mbar));
                mbar_arrive_expect_tx(kv_mbar, kKVTileBytes);
                uint32_t sKa = static_cast<uint32_t>(__cvta_generic_to_shared(sKbuf));
                uint32_t sVa = static_cast<uint32_t>(__cvta_generic_to_shared(sVbuf));
                // K: two D-halves (globalDim={kD, NH, S_total, n_pool}; box={64,1,kBN,1})
                tma_load_4d(sKa,               params.tma_K_cpu, 0,  head, n_start, pool_batch, mbar_addr);
                tma_load_4d(sKa + kTmaKVHalf,  params.tma_K_cpu, 64, head, n_start, pool_batch, mbar_addr);
                // V: 16 atoms (globalDim={kD,NH,S_total,n_pool}; box={64,1,8,1})
                // atom(d_o, s_o) smem offset = (d_o + s_o*2) * kVAtomBytes
                // global coord = {d_o*64, head, n_start+s_o*8, pool_batch}
                for (int s_o = 0; s_o < kVNSAtoms; ++s_o) {
                    for (int d_o = 0; d_o < 2; ++d_o) {
                        uint32_t voff = (uint32_t)((d_o + s_o * 2) * kVAtomBytes);
                        tma_load_4d(sVa + voff, params.tma_V_cpu,
                            d_o * 64, head, n_start + s_o * kVAtomS, pool_batch, mbar_addr);
                    }
                }
            }
            // All 256 threads wait for the tile to arrive.
            // mbar_wait with .shared::cta scope provides proxy-acquire fence (SM90).
            mbar_wait(kv_mbar, kv_phase);
            kv_phase ^= 1;

        } else {
            // New KV tile: project X[n_block] → sKbuf[cur] / sVbuf[cur]
            // Wk/Wv are already in persistent smem sWk_full/sWv_full.
            // ── WG0: X TMA producer ──────────────────────────────────────
            if (wg_idx == 0) {
                for (int cs = 0; cs < kNumProjChunks; ++cs) {
                    if (tid == 0) {
                        pipeline_proj.producer_acquire(pipe_proj_write);
                        int h_off    = cs * kHiddenChunk;
                        int stg      = pipe_proj_write.index();
                        uint32_t bar_addr = static_cast<uint32_t>(
                            __cvta_generic_to_shared(
                                pipeline_proj.producer_get_barrier(pipe_proj_write)));
                        uint32_t sx_addr = static_cast<uint32_t>(
                            __cvta_generic_to_shared(sX_p[stg]));
                        // X covers only new tokens: subtract seqlen_past to get local index.
                        tma_load_3d(sx_addr, params.tma_X, h_off, n_start - params.seqlen_past, batch, bar_addr);
                    }
                    ++pipe_proj_write;
                }
            } else {
                // ── WG1: WGMMA consumer → acc_k, acc_v ─────────────────
                TiledMmaProj tiled_mma_proj;
                auto thr = tiled_mma_proj.get_thread_slice(ctid);
                Tensor acc_k = partition_fragment_C(
                    tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
                Tensor acc_v = partition_fragment_C(
                    tiled_mma_proj, Shape<Int<kBlockN>, Int<kHeadDim>>{});
                clear(acc_k);
                clear(acc_v);

                for (int cs = 0; cs < kNumProjChunks; ++cs) {
                    pipeline_proj.consumer_wait(pipe_proj_read);
                    int stg = pipe_proj_read.index();

                    Tensor sX_cur  = make_tensor(make_smem_ptr(sX_p[stg]),  SmemLayoutX{});
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
                    cute::warpgroup_fence_operand(acc_k);
                    cute::warpgroup_fence_operand(acc_v);
                    pipeline_proj.consumer_release(pipe_proj_read);
                    ++pipe_proj_read;
                }

                // Write projected K,V to single smem buffer
                Tensor sK_cur = make_tensor(make_smem_ptr(sKbuf), SmemLayoutK{});
                Tensor sV_cur = make_tensor(make_smem_ptr(sVbuf), SmemLayoutV{});
                Tensor tCsK   = thr.partition_C(sK_cur);
                Tensor cV     = cute::make_identity_tensor(
                    Shape<Int<kBlockN>, Int<kHeadDim>>{});
                Tensor tCcV   = thr.partition_C(cV);
                #pragma unroll
                for (int i = 0; i < size(acc_k); ++i) {
                    tCsK(i) = Element(acc_k(i));
                    sV_cur(get<1>(tCcV(i)), get<0>(tCcV(i))) = Element(acc_v(i));
                }

                // K RoPE rotation (Neox style): applied in smem after WGMMA write,
                // before SYNC A so both the TMA store and QK GEMM see rotated K.
                // 128 WG1 threads cover kBlockN*half_D pairs at 32 pairs/thread.
                if (params.rope_mode == 1) {
                    constexpr int half_D = kHeadDim / 2;
                    #pragma unroll 1
                    for (int pair = ctid; pair < kBlockN * half_D; pair += kNThreadsMMA) {
                        int tok_local  = pair / half_D;
                        int d          = pair % half_D;
                        int global_pos = n_start + tok_local;
                        // Row p of cos_sin_cache: [cos_0..cos_{half_D-1}, sin_0..sin_{half_D-1}]
                        float cos_val = params.ptr_cos_sin[global_pos * kHeadDim + d];
                        float sin_val = params.ptr_cos_sin[global_pos * kHeadDim + half_D + d];
                        float k_lo = (float)sK_cur(tok_local, d);
                        float k_hi = (float)sK_cur(tok_local, d + half_D);
                        sK_cur(tok_local, d)          = Element(k_lo * cos_val - k_hi * sin_val);
                        sK_cur(tok_local, d + half_D) = Element(k_hi * cos_val + k_lo * sin_val);
                    }
                }
            }

            // SYNC A: sKbuf[cur] / sVbuf[cur] visible to all threads and
            // ready for TMA stores (which read smem) and WGMMA (which reads smem).
            __syncthreads();

            // Issue async write-back to CPU (tid==0 only).
            // tma_store_fence() makes prior WGMMA smem writes visible to the
            // TMA store engine before the bulk stores are enqueued.
            if (tid == 0) {
                tma_store_fence();
                uint32_t sKa = static_cast<uint32_t>(__cvta_generic_to_shared(sKbuf));
                uint32_t sVa = static_cast<uint32_t>(__cvta_generic_to_shared(sVbuf));
                // K write-back: two D-halves → K_cpu[pool_batch, n_start, head, 0..128]
                tma_store_4d(params.tma_K_cpu, 0,  head, n_start, pool_batch, sKa);
                tma_store_4d(params.tma_K_cpu, 64, head, n_start, pool_batch, sKa + kTmaKVHalf);
                // V write-back: 16 atoms → V_cpu [n_pool,S_total,NH,kD] token-major
                // global coord = {d_o*64, head, n_start+s_o*8, pool_batch}
                for (int s_o = 0; s_o < kVNSAtoms; ++s_o) {
                    for (int d_o = 0; d_o < 2; ++d_o) {
                        uint32_t voff = (uint32_t)((d_o + s_o * 2) * kVAtomBytes);
                        tma_store_4d(params.tma_V_cpu,
                            d_o * 64, head, n_start + s_o * kVAtomS, pool_batch, sVa + voff);
                    }
                }
                cp_async_bulk_commit_group();   // group G: this tile's write-back
            }
            // WG1 begins Q-inner immediately; NVLink-C2C write-back drains concurrently.
        }

        // ==============================================================
        // STEP 2: Q-inner attention loop (GQA-compatible: gqa_ratio g-iterations)
        // Uses sKbuf and sVbuf produced above; same K/V tile reused for all g.
        // wg0 enqueues all Q tiles across g, wg1 consumes them in the same order.
        // ==============================================================
        const int m_end = params.num_m_blocks;

        if (wg_idx == 0) {
            // Q TMA producer: iterate all gqa_ratio Q-head groups, then all m-blocks.
            // q_head = head * gqa_ratio + g indexes into Q[B, S_q, NH_q, D].
            for (int g = 0; g < gqa_ratio; ++g) {
                const int q_head = head * gqa_ratio + g;
                for (int m_block = 0; m_block < m_end; ++m_block) {
                    if (tid == 0) {
                        pipeline_q.producer_acquire(pipe_q_write);
                        int m_start = m_block * kBlockM;
                        int stg     = pipe_q_write.index();
                        Element* sq = (stg == 0) ? sQ0_ptr : sQ1_ptr;
                        uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(sq));
                        uint32_t s1 = s0 + kTmaQHalf;
                        uint32_t mb = static_cast<uint32_t>(
                            __cvta_generic_to_shared(
                                pipeline_q.producer_get_barrier(pipe_q_write)));
                        tma_load_4d(s0, params.tma_Q, 0,  q_head, m_start, batch, mb);
                        tma_load_4d(s1, params.tma_Q, 64, q_head, m_start, batch, mb);
                    }
                    ++pipe_q_write;
                }
            }
            // Drain CPU write-back (new-KV tiles): all g done, sKbuf/sVbuf safe to reuse.
            if (!is_past && tid == 0) {
                cp_async_bulk_wait_group<0>();
            }

        } else {
            // WG1: attention consumer — outer g-loop, inner m_block loop.
            TiledMmaQK tiled_mma_qk;
            TiledMmaPV tiled_mma_pv;
            auto thr_mma_qk = tiled_mma_qk.get_thread_slice(ctid);
            auto thr_mma_pv = tiled_mma_pv.get_thread_slice(ctid);

            // sKbuf/sVbuf are constant across all g for this n_block.
            Tensor sK_cur = make_tensor(make_smem_ptr(sKbuf), SmemLayoutK{});
            Tensor sV_cur = make_tensor(make_smem_ptr(sVbuf), SmemLayoutV{});

            Tensor sQ0_t  = make_tensor(make_smem_ptr(sQ0_ptr), SmemLayoutQ{});
            Tensor sQ1_t  = make_tensor(make_smem_ptr(sQ1_ptr), SmemLayoutQ{});
            Tensor tSrQ0  = thr_mma_qk.partition_fragment_A(sQ0_t);
            Tensor tSrQ1  = thr_mma_qk.partition_fragment_A(sQ1_t);
            Tensor tSrK   = thr_mma_qk.partition_fragment_B(sK_cur);
            Tensor tOrV   = thr_mma_pv.partition_fragment_B(sV_cur);

            auto smem_tiled_copy_O = make_tiled_copy_C(
                cute::Copy_Atom<cute::SM90_U32x4_STSM_N, Element>{}, tiled_mma_pv);
            auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(ctid);
            Tensor taccOsO0 = smem_thr_copy_O.partition_D(sO0);
            Tensor taccOsO1 = smem_thr_copy_O.partition_D(sO1);

            for (int g = 0; g < gqa_ratio; ++g) {
                // q_head: which Q-head this CTA computes for this g-iteration.
                // O_run/LSE_run are indexed by NH_q and q_head (different per g → no aliasing).
                const int q_head = head * gqa_ratio + g;

                for (int m_block = 0; m_block < m_end; ++m_block) {
                    const int m_start = m_block * kBlockM;
                    pipeline_q.consumer_wait(pipe_q_read);
                    int buf = pipe_q_read.index();

                    // Q RoPE: only when q_pos_start >= 0 (SGLang has NOT pre-rotated Q).
                    if (params.rope_mode == 1 && params.q_pos_start >= 0) {
                        Element* sQ_ptr_cur = (buf == 0) ? sQ0_ptr : sQ1_ptr;
                        Tensor sQ_rot = make_tensor(make_smem_ptr(sQ_ptr_cur), SmemLayoutQ{});
                        constexpr int half_D = kHeadDim / 2;
                        #pragma unroll 1
                        for (int pair = ctid; pair < kBlockM * half_D; pair += kNThreadsMMA) {
                            int tok_local  = pair / half_D;
                            int d          = pair % half_D;
                            int global_pos = params.q_pos_start + m_start + tok_local;
                            float cos_val  = params.ptr_cos_sin[global_pos * kHeadDim + d];
                            float sin_val  = params.ptr_cos_sin[global_pos * kHeadDim + half_D + d];
                            float q_lo = (float)sQ_rot(tok_local, d);
                            float q_hi = (float)sQ_rot(tok_local, d + half_D);
                            sQ_rot(tok_local, d)          = Element(q_lo * cos_val - q_hi * sin_val);
                            sQ_rot(tok_local, d + half_D) = Element(q_hi * cos_val + q_lo * sin_val);
                        }
                        cutlass::arch::NamedBarrier::sync(
                            static_cast<uint32_t>(kNThreadsMMA),
                            static_cast<uint32_t>(SmpV2Barrier::QRotReady));
                    }

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
                        int causal_row_offset = m_start + 1 + params.seqlen_past - n_start - thread_col_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(acc_s_rc); ++m) {
                            int row_rel   = get<0>(tScS_rc(m, _0{}));
                            int col_limit = row_rel + causal_row_offset;
                            #pragma unroll
                            for (int n = 0; n < size<1>(acc_s_rc); ++n) {
                                int col_rel_t0 = get<1>(t0ScS_rc(_0{}, n));
                                if (col_rel_t0 >= col_limit) acc_s_rc(m, n) = -INFINITY;
                            }
                        }
                    }

                    // Softmax
                    flash::Softmax<kNRows, 0> softmax(softmax_scale_log2);
                    (void)softmax.template max_get_scale<true, true>(acc_s);
                    softmax.template online_softmax<true, true>(acc_s);

                    // PV GEMM
                    Tensor tOrP_acc = make_tensor(acc_s.data(),
                        flash::convert_layout_acc_Aregs<TiledMmaPV>(acc_s.layout()));
                    Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
                    flash::convert_type_out(tOrP_acc, tOrP);

                    Tensor acc_o = partition_fragment_C(
                        tiled_mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});
                    flash::gemm<true, -1>(tiled_mma_pv, tOrP, tOrV, acc_o);

                    auto scores_scale = softmax.finalize();
                    cute::warpgroup_wait<0>();
                    cute::warpgroup_fence_operand(acc_o);
                    softmax.rescale_o(acc_o, scores_scale);

                    // Write LSE_local to smem
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

                    // STSM acc_o → sO (ping-pong)
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

                    // Sync: sO and sLSE_local ready for merge
                    cutlass::arch::NamedBarrier::sync(
                        static_cast<uint32_t>(kNThreadsMMA),
                        static_cast<uint32_t>(SmpV2Barrier::OSmemReady));

                    // LSE-form merge into HBM O_run/LSE_run.
                    // Row index uses NH_q (total Q-heads) and q_head (this g's Q-head).
                    {
                        Element* sO_cur_p = ((m_block & 1) == 0) ? sO0_ptr : sO1_ptr;
                        Tensor sO_cur = make_tensor(make_smem_ptr(sO_cur_p), SmemLayoutO{});
                        int row_in_tile = ctid >> 1;
                        int d_half      = ctid & 1;
                        int s_global    = m_start + row_in_tile;

                        if (s_global < S_q) {
                            int row_id = (batch * S_q + s_global) * NH_q + q_head;

                            float LSE_local_val = sLSE_local[row_in_tile];
                            float LSE_old       = LSE_run_ptr[row_id];

                            float alpha, beta, LSE_new;
                            float LSE_max = fmaxf(LSE_old, LSE_local_val);
                            if (LSE_max == -INFINITY) goto skip_merge;
                            {
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

                            {
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
                            skip_merge:;
                        }

                        cutlass::arch::NamedBarrier::sync(
                            static_cast<uint32_t>(kNThreadsMMA),
                            static_cast<uint32_t>(SmpV2Barrier::MergeStatsReady));
                    }
                }   // end m_block loop
            }   // end g loop (GQA Q-head groups)
        }       // end wg_idx == 1

        // SYNC B: both wg0 and wg1 done with all g iterations; write-back drained.
        // sKbuf/sVbuf are safe to overwrite in the next n_block iteration.
        __syncthreads();

    }   // end n_block loop

    // ------------------------------------------------------------------
    // FINALIZE: cast O_run (fp32) → O (bf16).
    // GQA: this CTA owns gqa_ratio Q-heads; write them all out.
    // O layout: [B, S_q, NH_q, kD] (same for MHA where gqa_ratio=1, NH_q=NH).
    // ------------------------------------------------------------------
    __syncthreads();
    {
        int total = S_q * kHeadDim;
        for (int g = 0; g < gqa_ratio; ++g) {
            const int q_head = head * gqa_ratio + g;
            for (int idx = tid; idx < total; idx += 256) {
                int s = idx / kHeadDim;
                int d = idx % kHeadDim;
                int row_id = (batch * S_q + s) * NH_q + q_head;
                float o_val = O_run_ptr[row_id * kHeadDim + d];
                size_t off  = (((size_t)batch * S_q + s) * NH_q + q_head) * kHeadDim + d;
                O_final_ptr[off] = (Element)o_val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Smem size helper
// ---------------------------------------------------------------------------
template <typename Traits>
size_t sm_parallel_v2_cpu_kv_smem_bytes()
{
    using Element = typename Traits::Element;

    using SmemLayoutAtomW_full = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element,
            cute::Int<Traits::kHeadDim>, cute::Int<Traits::kHiddenDim>>());
    using SmemLayoutW_full = decltype(cute::tile_to_shape(
        SmemLayoutAtomW_full{},
        cute::make_shape(cute::Int<Traits::kHeadDim>{},
                         cute::Int<Traits::kHiddenDim>{})));

    auto r128 = [](size_t s) { return (s + 127) & ~size_t(127); };
    size_t smem = 128;
    smem += r128(cute::cosize(typename Traits::SmemLayoutQ{}) * sizeof(Element));  // sQ0
    smem += r128(cute::cosize(typename Traits::SmemLayoutQ{}) * sizeof(Element));  // sQ1
    smem += r128(cute::cosize(SmemLayoutW_full{}) * sizeof(Element));              // sWk_full
    smem += r128(cute::cosize(SmemLayoutW_full{}) * sizeof(Element));              // sWv_full
    smem += r128(cute::cosize(typename Traits::SmemLayoutK{}) * sizeof(Element));  // sKbuf
    smem += r128(cute::cosize(typename Traits::SmemLayoutV{}) * sizeof(Element));  // sVbuf
    smem += r128(cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));  // sO0
    smem += r128(cute::cosize(typename Traits::SmemLayoutO{}) * sizeof(Element));  // sO1
    smem += 8;    // startup_mbar
    smem += 8;    // kv_mbar
    smem += sizeof(PipelineProj::SharedStorage) + 16;
    smem += sizeof(PipelineQ::SharedStorage)    + 16;
    smem += Traits::kBlockM * sizeof(float) + 16;   // sLSE_local
    return smem;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
template <typename Traits>
cudaError_t launch_sm_parallel_v2_cpu_kv(
    SmParallelV2CpuKVParams p, cudaStream_t stream = 0)
{
    size_t smem = sm_parallel_v2_cpu_kv_smem_bytes<Traits>();
    auto* kernel = &sm_parallel_v2_cpu_kv_fwd_kernel<Traits>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (err != cudaSuccess) return err;
    dim3 grid(p.num_heads, p.batch);
    kernel<<<grid, 256, smem, stream>>>(p);
    return cudaGetLastError();
}

} // namespace sm_parallel_v2_cpu_kv
