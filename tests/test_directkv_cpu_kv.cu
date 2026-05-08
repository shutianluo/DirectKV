/******************************************************************************
 * Correctness test — DirectKV CPU KV Cache kernel
 *
 * Compile (from tests/ directory):
 *   nvcc -O2 -arch=sm_90a -std=c++17 --expt-relaxed-constexpr \
 *        -I/path/to/cutlass/include \
 *        test_directkv_cpu_kv.cu -o test_directkv_cpu_kv \
 *        -lcuda
 *
 * Tests:
 *   Prefill (all new KV): verify O, K_cpu write-back, V_cpu write-back
 *   Decode  (past + new): verify O, new K_cpu append, past K_cpu preserved
 ******************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <cassert>

// Kernel
#include "../csrc/directKV_kernel.cuh"

using BF16 = __nv_bfloat16;
using Traits = proj_fused::DefaultTraits_hdim128_bf16;

constexpr int kBN  = Traits::kBlockN;   // 64
constexpr int kBM  = Traits::kBlockM;   // 64
constexpr int kD   = Traits::kHeadDim;  // 128
constexpr int kHC  = Traits::kHiddenChunk; // 64

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } } while(0)

#define CU_CHECK(e) do { \
    CUresult _e = (e); \
    if (_e != CUDA_SUCCESS) { \
        const char* s; cuGetErrorString(_e, &s); \
        fprintf(stderr, "CU error %s at %s:%d\n", s, __FILE__, __LINE__); \
        exit(1); \
    } } while(0)

// ---------------------------------------------------------------------------
// TMA descriptor helpers (bf16 variants)
// ---------------------------------------------------------------------------

// K cache: [B, S_total, NH, D] row-major → 4D: (D, NH, S, B), box={64, 1, kBN, 1}
inline CUresult create_tma_K_cpu(CUtensorMap* desc, void const* ptr,
    int B, int S_total, int NH, int D, int kBlockN)
{
    uint64_t gd[4] = {(uint64_t)D, (uint64_t)NH, (uint64_t)S_total, (uint64_t)B};
    uint64_t gs[3] = {
        (uint64_t)D  * 2,
        (uint64_t)NH * D * 2,
        (uint64_t)S_total * NH * D * 2
    };
    uint32_t bx[4] = {64, 1, (uint32_t)kBlockN, 1};
    uint32_t es[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        const_cast<void*>(ptr), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// V cache: [B, S_total, NH, D] — D-fastest (globalDim[0]=D), same as K.
// Box={64, 1, 8, 1}: D=64 (128B/row, matches 128B swizzle) × NH=1 × S=8 = one MN-major
// smem atom [kD_atom=64, kBN_atom=8].  16 TMA ops/tile at smem offset
// (d_outer + s_outer*2)*1024 bytes; global coord = {d_o*64, head, n_start+s_o*8, batch}.
// NOTE: 128B swizzle requires the fastest box dim = 64 elems × 2 bytes = 128B/row.
//       MN-major global [B,NH,D,S] (S-fastest) would give 8-elem rows (16B) → swizzle mismatch.
//       Token-major [B,S,NH,D] (D-fastest) keeps D as box[0]=64 → correct swizzle mapping.
inline CUresult create_tma_V_cpu(CUtensorMap* desc, void const* ptr,
    int B, int S_total, int NH, int D)
{
    uint64_t gd[4] = {(uint64_t)D, (uint64_t)NH, (uint64_t)S_total, (uint64_t)B};
    uint64_t gs[3] = {
        (uint64_t)D  * 2,
        (uint64_t)NH * D * 2,
        (uint64_t)S_total * NH * D * 2
    };
    uint32_t bx[4] = {64, 1, 8, 1};   // kD_atom=64, kBN_atom=8
    uint32_t es[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        const_cast<void*>(ptr), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// Q: [B, S_q, NH, D] → 4D: (D, NH, S, B), box={64, 1, kBM, 1}
inline CUresult create_tma_Q_bf16(CUtensorMap* desc, void const* ptr,
    int B, int S, int NH, int D)
{
    uint64_t gd[4] = {(uint64_t)D, (uint64_t)NH, (uint64_t)S, (uint64_t)B};
    uint64_t gs[3] = {(uint64_t)D*2, (uint64_t)NH*D*2, (uint64_t)S*NH*D*2};
    uint32_t bx[4] = {64, 1, 64, 1};
    uint32_t es[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        const_cast<void*>(ptr), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// X: [B, S, H] → 3D: (H, S, B), box={kHC, kBN, 1}
inline CUresult create_tma_X_bf16(CUtensorMap* desc, void const* ptr,
    int B, int S, int H, int kHC_, int kBlockN)
{
    uint64_t gd[3] = {(uint64_t)H, (uint64_t)S, (uint64_t)B};
    uint64_t gs[2] = {(uint64_t)H*2, (uint64_t)S*H*2};
    uint32_t bx[3] = {(uint32_t)kHC_, (uint32_t)kBlockN, 1};
    uint32_t es[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
        const_cast<void*>(ptr), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// W: [NH, D, H] → 3D: (H, D, NH), box={kHC, D, 1}
inline CUresult create_tma_W_bf16(CUtensorMap* desc, void const* ptr,
    int NH, int D, int H, int kHC_)
{
    uint64_t gd[3] = {(uint64_t)H, (uint64_t)D, (uint64_t)NH};
    uint64_t gs[2] = {(uint64_t)H*2, (uint64_t)D*H*2};
    uint32_t bx[3] = {(uint32_t)kHC_, (uint32_t)D, 1};
    uint32_t es[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
        const_cast<void*>(ptr), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// ---------------------------------------------------------------------------
// Host reference (fp32)
// ---------------------------------------------------------------------------
// Reference: for a single (batch, head), compute attention output
// Q[S_q, D], K[S_kv, D], V[S_kv, D] (all fp32), is_causal, s_past = number of past KV
static void ref_attention(
    float* O, const float* Q, const float* K, const float* V,
    int S_q, int S_kv, int D, float scale, bool is_causal, int s_past)
{
    std::vector<float> scores(S_q * S_kv);
    // scores = Q @ K^T * scale
    for (int i = 0; i < S_q; ++i)
        for (int j = 0; j < S_kv; ++j) {
            float s = 0;
            for (int d = 0; d < D; ++d) s += Q[i*D+d] * K[j*D+d];
            scores[i*S_kv+j] = s * scale;
        }
    // Causal mask: query at global position (s_past + i) can only attend to j <= s_past + i
    if (is_causal) {
        for (int i = 0; i < S_q; ++i)
            for (int j = 0; j < S_kv; ++j)
                if (j > s_past + i) scores[i*S_kv+j] = -1e30f;
    }
    // Softmax over j for each i, then weighted sum
    for (int i = 0; i < S_q; ++i) {
        float maxv = -1e30f;
        for (int j = 0; j < S_kv; ++j) maxv = fmaxf(maxv, scores[i*S_kv+j]);
        float sumv = 0;
        for (int j = 0; j < S_kv; ++j) { scores[i*S_kv+j] = expf(scores[i*S_kv+j]-maxv); sumv += scores[i*S_kv+j]; }
        for (int j = 0; j < S_kv; ++j) scores[i*S_kv+j] /= sumv;
        for (int d = 0; d < D; ++d) {
            float o = 0;
            for (int j = 0; j < S_kv; ++j) o += scores[i*S_kv+j] * V[j*D+d];
            O[i*D+d] = o;
        }
    }
}

// ---------------------------------------------------------------------------
// Fill a buffer with random bf16 values scaled by `scale`
// ---------------------------------------------------------------------------
static void fill_random_bf16(BF16* p, size_t n, float scale, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i)
        p[i] = __float2bfloat16((float(rand()) / RAND_MAX - 0.5f) * 2.0f * scale);
}

// ---------------------------------------------------------------------------
// Max abs error between float arrays
// ---------------------------------------------------------------------------
static float max_abs_error(const float* a, const float* b, size_t n) {
    float e = 0;
    for (size_t i = 0; i < n; ++i) e = fmaxf(e, fabsf(a[i]-b[i]));
    return e;
}

// ---------------------------------------------------------------------------
// Test configuration
// ---------------------------------------------------------------------------
struct Config {
    const char* name;
    int B, NH, S_new, S_past, H;
    bool is_causal;
};

// ---------------------------------------------------------------------------
// Run one test
// ---------------------------------------------------------------------------
static bool run_test(const Config& cfg) {
    const int S_total = cfg.S_past + cfg.S_new;
    const int S_q     = cfg.S_new;          // Q sequence length = new tokens
    const int NH = cfg.NH, B = cfg.B, H = cfg.H;
    const float scale = 1.0f / sqrtf((float)kD);

    assert(cfg.S_past % kBN == 0 && "seqlen_past must be multiple of kBN");
    assert(cfg.S_new  % kBN == 0 && "seqlen_new must be multiple of kBN");
    assert(S_q % kBM  == 0 && "S_q must be multiple of kBM");

    // Sizes
    size_t X_sz   = (size_t)B * S_q  * H   * sizeof(BF16);
    size_t Wk_sz  = (size_t)NH * kD  * H   * sizeof(BF16);
    size_t Wv_sz  = (size_t)NH * kD  * H   * sizeof(BF16);
    size_t Q_sz   = (size_t)B * S_q  * NH  * kD * sizeof(BF16);
    size_t O_sz   = Q_sz;
    size_t Or_sz  = (size_t)B * S_q  * NH  * kD * sizeof(float);
    size_t Lr_sz  = (size_t)B * S_q  * NH  * sizeof(float);
    size_t Kcpu_sz = (size_t)B * S_total * NH * kD * sizeof(BF16);   // K: [B,S_total,NH,D] D-fastest
    size_t Vcpu_sz = (size_t)B * S_total * NH * kD * sizeof(BF16);   // V: [B,S_total,NH,D] D-fastest

    // Host CPU pinned buffers
    BF16 *K_cpu, *V_cpu;
    CUDA_CHECK(cudaHostAlloc(&K_cpu, Kcpu_sz, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&V_cpu, Vcpu_sz, cudaHostAllocMapped));
    memset(K_cpu, 0, Kcpu_sz);
    memset(V_cpu, 0, Vcpu_sz);

    // Host input arrays
    std::vector<BF16> h_X(B*S_q*H), h_Wk(NH*kD*H), h_Wv(NH*kD*H), h_Q(B*S_q*NH*kD);
    fill_random_bf16(h_X.data(), h_X.size(), 0.1f, 42);
    fill_random_bf16(h_Wk.data(), h_Wk.size(), 0.1f, 43);
    fill_random_bf16(h_Wv.data(), h_Wv.size(), 0.1f, 44);
    fill_random_bf16(h_Q.data(), h_Q.size(), 0.1f, 45);

    // If decode: pre-populate past KV in pinned memory
    std::vector<float> K_past_ref, V_past_ref;
    if (cfg.S_past > 0) {
        K_past_ref.resize((size_t)B * cfg.S_past * NH * kD);
        V_past_ref.resize((size_t)B * cfg.S_past * NH * kD);
        // Random fp32 past KV (converted to bf16 for storage)
        srand(99);
        for (auto& v : K_past_ref) v = (float(rand())/RAND_MAX - 0.5f)*0.2f;
        for (auto& v : V_past_ref) v = (float(rand())/RAND_MAX - 0.5f)*0.2f;

        // Write to K_cpu[b, 0..S_past, h, d] = [B, S_total, NH, D]
        for (int b = 0; b < B; ++b)
        for (int s = 0; s < cfg.S_past; ++s)
        for (int h = 0; h < NH; ++h)
        for (int d = 0; d < kD; ++d) {
            size_t ki = ((size_t)b*S_total + s)*NH*kD + (size_t)h*kD + d;
            size_t ri = ((size_t)b*cfg.S_past + s)*NH*kD + (size_t)h*kD + d;
            K_cpu[ki] = __float2bfloat16(K_past_ref[ri]);
        }
        // Write to V_cpu[b, 0..S_past, h, d] = [B, S_total, NH, D] token-major (D fastest)
        for (int b = 0; b < B; ++b)
        for (int s = 0; s < cfg.S_past; ++s)
        for (int h = 0; h < NH; ++h)
        for (int d = 0; d < kD; ++d) {
            size_t vi = ((size_t)b*S_total + s)*NH*kD + (size_t)h*kD + d;
            size_t ri = ((size_t)b*cfg.S_past + s)*NH*kD + (size_t)h*kD + d;
            V_cpu[vi] = __float2bfloat16(V_past_ref[ri]);
        }
    }

    // GPU allocations
    BF16  *d_X, *d_Wk, *d_Wv, *d_Q, *d_O;
    float *d_O_run, *d_LSE_run;
    CUDA_CHECK(cudaMalloc(&d_X,      X_sz));
    CUDA_CHECK(cudaMalloc(&d_Wk,     Wk_sz));
    CUDA_CHECK(cudaMalloc(&d_Wv,     Wv_sz));
    CUDA_CHECK(cudaMalloc(&d_Q,      Q_sz));
    CUDA_CHECK(cudaMalloc(&d_O,      O_sz));
    CUDA_CHECK(cudaMalloc(&d_O_run,  Or_sz));
    CUDA_CHECK(cudaMalloc(&d_LSE_run, Lr_sz));

    CUDA_CHECK(cudaMemcpy(d_X,  h_X.data(),  X_sz,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk, h_Wk.data(), Wk_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv, h_Wv.data(), Wv_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q,  h_Q.data(),  Q_sz,  cudaMemcpyHostToDevice));
    // Init O_run=0, LSE_run=-INF
    CUDA_CHECK(cudaMemset(d_O_run, 0, Or_sz));
    {
        std::vector<float> lse_init(B*S_q*NH, -FLT_MAX);
        CUDA_CHECK(cudaMemcpy(d_LSE_run, lse_init.data(), Lr_sz, cudaMemcpyHostToDevice));
    }

    // TMA descriptors on device
    CUtensorMap *d_tma_X, *d_tma_Wk, *d_tma_Wv, *d_tma_Q, *d_tma_K_cpu, *d_tma_V_cpu;
    CUDA_CHECK(cudaMalloc(&d_tma_X,     sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&d_tma_Wk,    sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&d_tma_Wv,    sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&d_tma_Q,     sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&d_tma_K_cpu, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&d_tma_V_cpu, sizeof(CUtensorMap)));

    CUtensorMap h_tma_X, h_tma_Wk, h_tma_Wv, h_tma_Q, h_tma_K, h_tma_V;
    CU_CHECK(create_tma_X_bf16(&h_tma_X, d_X, B, S_q, H, kHC, kBN));
    CU_CHECK(create_tma_W_bf16(&h_tma_Wk, d_Wk, NH, kD, H, kHC));
    CU_CHECK(create_tma_W_bf16(&h_tma_Wv, d_Wv, NH, kD, H, kHC));
    CU_CHECK(create_tma_Q_bf16(&h_tma_Q, d_Q, B, S_q, NH, kD));
    CU_CHECK(create_tma_K_cpu(&h_tma_K, K_cpu, B, S_total, NH, kD, kBN));
    CU_CHECK(create_tma_V_cpu(&h_tma_V, V_cpu, B, S_total, NH, kD));

    CUDA_CHECK(cudaMemcpy(d_tma_X,     &h_tma_X,  sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma_Wk,    &h_tma_Wk, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma_Wv,    &h_tma_Wv, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma_Q,     &h_tma_Q,  sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma_K_cpu, &h_tma_K,  sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma_V_cpu, &h_tma_V,  sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    // Build params
    sm_parallel_v2_cpu_kv::SmParallelV2CpuKVParams p{};
    p.ptr_X        = d_X;
    p.ptr_Wk       = d_Wk;
    p.ptr_Wv       = d_Wv;
    p.ptr_Q        = d_Q;
    p.ptr_O        = d_O;
    p.ptr_O_run    = d_O_run;
    p.ptr_LSE_run  = d_LSE_run;
    p.ptr_K_cpu    = K_cpu;
    p.ptr_V_cpu    = V_cpu;
    p.batch        = B;
    p.seqlen_new   = S_q;
    p.seqlen_q     = S_q;
    p.num_heads    = NH;
    p.head_dim     = kD;
    p.hidden_dim   = H;
    p.seqlen_past  = cfg.S_past;
    p.num_n_blocks_past = cfg.S_past / kBN;
    p.num_n_blocks      = S_total / kBN;
    p.num_m_blocks      = S_q / kBM;
    p.softmax_scale     = scale;
    p.is_causal         = cfg.is_causal ? 1 : 0;
    p.num_q_heads       = NH;   // MHA: Q-heads == KV-heads
    p.gqa_ratio         = 1;    // MHA: one Q-head per KV-head
    p.tma_X    = d_tma_X;
    p.tma_Wk   = d_tma_Wk;
    p.tma_Wv   = d_tma_Wv;
    p.tma_Q    = d_tma_Q;
    p.tma_K_cpu = d_tma_K_cpu;
    p.tma_V_cpu = d_tma_V_cpu;

    // Launch
    CUDA_CHECK(sm_parallel_v2_cpu_kv::launch_sm_parallel_v2_cpu_kv<Traits>(p, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // Reference computation (fp32, host)
    // ------------------------------------------------------------------
    // 1. Project new tokens: K_new_ref[b, s, h, d] = X[b, s, :] @ Wk[h, :, d]^T
    std::vector<float> X_fp32(B*S_q*H), Wk_fp32(NH*kD*H), Wv_fp32(NH*kD*H), Q_fp32(B*S_q*NH*kD);
    for (size_t i = 0; i < h_X.size();  ++i) X_fp32[i]  = __bfloat162float(h_X[i]);
    for (size_t i = 0; i < h_Wk.size(); ++i) Wk_fp32[i] = __bfloat162float(h_Wk[i]);
    for (size_t i = 0; i < h_Wv.size(); ++i) Wv_fp32[i] = __bfloat162float(h_Wv[i]);
    for (size_t i = 0; i < h_Q.size();  ++i) Q_fp32[i]  = __bfloat162float(h_Q[i]);

    // K_new_ref[b, s, h, :] = X[b, s, :] @ Wk[h, :, :]^T
    // Wk: [NH, kD, H] = [head, out_dim, in_dim]; X: [B, S, H]
    // K_new_ref: [B, S, NH, kD]
    std::vector<float> K_new_fp32(B*S_q*NH*kD, 0), V_new_fp32(B*S_q*NH*kD, 0);
    for (int b = 0; b < B; ++b)
    for (int h = 0; h < NH; ++h)
    for (int s = 0; s < S_q; ++s) {
        // X[b,s,:] @ Wk[h,:,:]^T  — Wk[h, d, :] = row d of this head's weight
        for (int d = 0; d < kD; ++d) {
            float sk = 0, sv = 0;
            for (int hh = 0; hh < H; ++hh) {
                sk += X_fp32[b*S_q*H + s*H + hh] * Wk_fp32[h*kD*H + d*H + hh];
                sv += X_fp32[b*S_q*H + s*H + hh] * Wv_fp32[h*kD*H + d*H + hh];
            }
            K_new_fp32[b*S_q*NH*kD + s*NH*kD + h*kD + d] = sk;
            V_new_fp32[b*S_q*NH*kD + s*NH*kD + h*kD + d] = sv;
        }
    }

    // Full KV: K_full[b, s, h, d] for s in [0, S_total)
    //   s < S_past: from K_past_ref / V_past_ref
    //   s >= S_past: from K_new_fp32 / V_new_fp32
    std::vector<float> K_full(B*S_total*NH*kD, 0), V_full(B*S_total*NH*kD, 0);
    if (cfg.S_past > 0) {
        for (int b=0; b<B; ++b) for (int s=0; s<cfg.S_past; ++s)
        for (int h=0; h<NH; ++h) for (int d=0; d<kD; ++d) {
            size_t fi = b*S_total*NH*kD + s*NH*kD + h*kD + d;
            size_t pi = b*cfg.S_past*NH*kD + s*NH*kD + h*kD + d;
            K_full[fi] = K_past_ref[pi];
            V_full[fi] = V_past_ref[pi];
        }
    }
    for (int b=0; b<B; ++b) for (int s=0; s<S_q; ++s)
    for (int h=0; h<NH; ++h) for (int d=0; d<kD; ++d) {
        size_t fi = b*S_total*NH*kD + (cfg.S_past+s)*NH*kD + h*kD + d;
        size_t ni = b*S_q*NH*kD + s*NH*kD + h*kD + d;
        K_full[fi] = K_new_fp32[ni];
        V_full[fi] = V_new_fp32[ni];
    }

    // Attention reference: O_ref[b, s_q, h, :] = attn(Q[b,s_q,h,:], K_full[b,:,h,:], V_full[b,:,h,:])
    std::vector<float> O_ref(B*S_q*NH*kD, 0);
    for (int b = 0; b < B; ++b)
    for (int h = 0; h < NH; ++h) {
        // Extract Q[b, :, h, :] → [S_q, kD]
        std::vector<float> Qbh(S_q*kD), Kbh(S_total*kD), Vbh(S_total*kD);
        for (int s=0; s<S_q; ++s)
            for (int d=0; d<kD; ++d)
                Qbh[s*kD+d] = Q_fp32[b*S_q*NH*kD + s*NH*kD + h*kD + d];
        for (int s=0; s<S_total; ++s)
            for (int d=0; d<kD; ++d) {
                Kbh[s*kD+d] = K_full[b*S_total*NH*kD + s*NH*kD + h*kD + d];
                Vbh[s*kD+d] = V_full[b*S_total*NH*kD + s*NH*kD + h*kD + d];
            }
        std::vector<float> Obh(S_q*kD);
        ref_attention(Obh.data(), Qbh.data(), Kbh.data(), Vbh.data(),
                      S_q, S_total, kD, scale, cfg.is_causal, cfg.S_past);
        for (int s=0; s<S_q; ++s)
            for (int d=0; d<kD; ++d)
                O_ref[b*S_q*NH*kD + s*NH*kD + h*kD + d] = Obh[s*kD+d];
    }

    // ------------------------------------------------------------------
    // Verify O output
    // ------------------------------------------------------------------
    std::vector<BF16> h_O(B*S_q*NH*kD);
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, O_sz, cudaMemcpyDeviceToHost));
    std::vector<float> O_got(B*S_q*NH*kD);
    for (size_t i=0; i<h_O.size(); ++i) O_got[i] = __bfloat162float(h_O[i]);
    float o_err = max_abs_error(O_got.data(), O_ref.data(), O_got.size());

    // ------------------------------------------------------------------
    // Verify K_cpu write-back (new tokens at [*, S_past..S_total, *, *])
    // ------------------------------------------------------------------
    float k_err = 0;
    for (int b=0; b<B; ++b) for (int s=0; s<S_q; ++s)
    for (int h=0; h<NH; ++h) for (int d=0; d<kD; ++d) {
        size_t ki = ((size_t)b*S_total + (cfg.S_past+s))*NH*kD + (size_t)h*kD + d;
        float ref = K_new_fp32[b*S_q*NH*kD + s*NH*kD + h*kD + d];
        float got = __bfloat162float(K_cpu[ki]);
        k_err = fmaxf(k_err, fabsf(got - ref));
    }

    // ------------------------------------------------------------------
    // Verify V_cpu write-back ([B,S_total,NH,D]: V_cpu[b, S_past+s, h, d])
    // ------------------------------------------------------------------
    float v_err = 0;
    for (int b=0; b<B; ++b) for (int s=0; s<S_q; ++s)
    for (int h=0; h<NH; ++h) for (int d=0; d<kD; ++d) {
        size_t vi = ((size_t)b*S_total + (cfg.S_past+s))*NH*kD + (size_t)h*kD + d;
        float ref = V_new_fp32[b*S_q*NH*kD + s*NH*kD + h*kD + d];
        float got = __bfloat162float(V_cpu[vi]);
        v_err = fmaxf(v_err, fabsf(got - ref));
    }

    // ------------------------------------------------------------------
    // Verify past K_cpu is unchanged (decode only)
    // ------------------------------------------------------------------
    float past_k_err = 0;
    if (cfg.S_past > 0) {
        for (int b=0; b<B; ++b) for (int s=0; s<cfg.S_past; ++s)
        for (int h=0; h<NH; ++h) for (int d=0; d<kD; ++d) {
            size_t ki = ((size_t)b*S_total + s)*NH*kD + (size_t)h*kD + d;
            float ref = K_past_ref[b*cfg.S_past*NH*kD + s*NH*kD + h*kD + d];
            float got = __bfloat162float(K_cpu[ki]);
            past_k_err = fmaxf(past_k_err, fabsf(got - ref));
        }
    }

    // Thresholds (bf16 tolerance)
    const float kOThresh    = 2e-2f;
    const float kKVThresh   = 1e-2f;
    const float kPastThresh = 5e-4f;  // bf16 round-trip tolerance for pre-populated past KV

    bool pass = (o_err < kOThresh) && (k_err < kKVThresh) &&
                (v_err < kKVThresh);

    printf("%-5s  O_err=%.5f(%s)  K_err=%.5f(%s)  V_err=%.5f(%s)  → %s\n",
        cfg.name,
        o_err,  o_err  < kOThresh  ? "OK" : "FAIL",
        k_err,  k_err  < kKVThresh ? "OK" : "FAIL",
        v_err,  v_err  < kKVThresh ? "OK" : "FAIL",
        pass ? "PASS" : "FAIL");

    // ------------------------------------------------------------------
    // Print O snapshot: batch=0, head=0, first kPrintTokens tokens × kPrintDims dims
    // Layout: O[b, s, h, d] = O[b*S_q*NH*kD + s*NH*kD + h*kD + d]
    // ------------------------------------------------------------------
    // {
    //     constexpr int kPrintTokens = 3;
    //     constexpr int kPrintDims   = 3;
    //     int print_s = std::min(kPrintTokens, S_q);
    //     int print_d = std::min(kPrintDims, kD);
    //     printf("  O snapshot [b=0 h=0 s=0..%d d=0..%d]:\n", print_s-1, print_d-1);
    //     printf("  %8s", "");
    //     for (int d = 0; d < print_d; ++d) printf("      d=%-4d", d);
    //     printf("\n");
    //     for (int s = 0; s < print_s; ++s) {
    //         size_t base = (size_t)s * NH * kD + 0 * kD;  // b=0, h=0
    //         printf("  ref  s=%-3d", s);
    //         for (int d = 0; d < print_d; ++d)
    //             printf("  %+.4f", O_ref[base + d]);
    //         printf("\n");
    //         printf("  ker  s=%-3d", s);
    //         for (int d = 0; d < print_d; ++d)
    //             printf("  %+.4f", O_got[base + d]);
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // Cleanup
    cudaFreeHost(K_cpu); cudaFreeHost(V_cpu);
    cudaFree(d_X); cudaFree(d_Wk); cudaFree(d_Wv); cudaFree(d_Q);
    cudaFree(d_O); cudaFree(d_O_run); cudaFree(d_LSE_run);
    cudaFree(d_tma_X); cudaFree(d_tma_Wk); cudaFree(d_tma_Wv);
    cudaFree(d_tma_Q); cudaFree(d_tma_K_cpu); cudaFree(d_tma_V_cpu);
    return pass;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    size_t smem = sm_parallel_v2_cpu_kv::sm_parallel_v2_cpu_kv_smem_bytes<Traits>();
    printf("SM-Parallel v2 CPU KV — Correctness Tests (GH200 NVLink-C2C)\n");
    printf("Traits: kBM=%d kBN=%d kD=%d kHC=%d  smem=%.1f KB\n\n",
           kBM, kBN, kD, kHC, smem / 1024.0);

    // Prefill configs (S_past=0, all new KV)
    Config prefill_tests[] = {
        // name               B  NH  S_new  S_past  H    causal
        {"pf-tiny",           1,  2,   64,      0, 128, false},
        {"pf-tiny-causal",    1,  2,   64,      0, 128, true },
        {"pf-small-causal",   1,  8,  512,      0, 128, true },
        {"pf-1k-causal",      1,  8, 1024,      0, 128, true },
        {"pf-1k-noncausal",   1,  8, 1024,      0, 128, false},
        {"pf-b4",             4,  8,  512,      0, 128, true },
    };

    // Decode configs (S_past>0, mixed past + new KV)
    Config decode_tests[] = {
        // name               B  NH  S_new  S_past  H    causal
        {"dec-tiny",          1,  2,   64,    128, 128, true },
        {"dec-small",         1,  8,   64,    512, 128, true },
        {"dec-medium",        1,  8,  128,   1024, 128, true },
        {"dec-noncausal",     1,  4,  128,    512, 128, false},
        {"dec-b4",            4,  8,   64,    256, 128, true },
        {"dec-longctx",       1,  8,   64,   4096, 128, true },
    };

    int passed = 0, total = 0;

    printf("── Prefill (all new KV) ──\n");
    for (auto& cfg : prefill_tests) {
        ++total;
        if (run_test(cfg)) ++passed;
    }

    printf("\n── Decode (past + new KV) ──\n");
    for (auto& cfg : decode_tests) {
        ++total;
        if (run_test(cfg)) ++passed;
    }

    printf("\n%d / %d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
