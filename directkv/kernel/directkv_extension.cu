/*
 * DirectKV CPU-KV CUDA Extension (SM90, bf16, GH200 NVLink-C2C)
 * ===========================================================
 * Pybind11 wrapper for sm_parallel_v2_cpu_kv_fwd_kernel.
 * Handles TMA descriptor creation, persistent buffer management, and
 * RoPE parameter wiring to SmParallelV2CpuKVParams.
 *
 * Python signature:
 *   directkv_fwd(X, Wk, Wv, Q, K_cpu, V_cpu, cos_sin,
 *             softmax_scale, is_causal, seqlen_past, q_pos_start) -> O
 *
 * Tensor layout (all bf16 unless noted):
 *   X       : [B, S_new, hidden_dim]     GPU
 *   Wk, Wv  : [NH, head_dim, hidden_dim] GPU
 *   Q       : [B, S_new, NH, head_dim]   GPU
 *   K_cpu   : [B, S_total, NH, head_dim] CPU-pinned   (S_total = S_past + S_new)
 *   V_cpu   : [B, S_total, NH, head_dim] CPU-pinned
 *   cos_sin : [max_pos, head_dim]         GPU fp32, or None (disables RoPE)
 *   O       : [B, S_new, NH, head_dim]   GPU bf16  (returned)
 *
 * Constraints:
 *   head_dim  == 128  (DefaultTraits_hdim128_bf16)
 *   seqlen_past % 64 == 0,  S_new % 64 == 0
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <limits>

// Main kernel (includes proj_fused_kernel_traits_sm90.h, softmax.h, utils.h)
#include "directKV_kernel.cuh"

#define SMPV2_CHECK_CUDA(x)   TORCH_CHECK((x).device().is_cuda(),  #x " must be a CUDA tensor")
#define SMPV2_CHECK_CPU(x)    TORCH_CHECK((x).device().is_cpu(),   #x " must be a CPU tensor")
#define SMPV2_CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(),    #x " must be contiguous")
#define SMPV2_CHECK_BF16(x)   TORCH_CHECK((x).scalar_type() == at::kBFloat16, #x " must be bf16")
#define SMPV2_CHECK_F32(x)    TORCH_CHECK((x).scalar_type() == at::kFloat,    #x " must be fp32")

using Traits = proj_fused::DefaultTraits_hdim128_bf16;

static constexpr int kBN = Traits::kBlockN;     // 64
static constexpr int kBM = Traits::kBlockM;     // 64
static constexpr int kD  = Traits::kHeadDim;    // 128
static constexpr int kHC = Traits::kHiddenChunk; // 64

// --------------------------------------------------------------------------
// Persistent device storage (allocated once per process)
// --------------------------------------------------------------------------
static CUtensorMap* g_tma_X     = nullptr;
static CUtensorMap* g_tma_Wk    = nullptr;
static CUtensorMap* g_tma_Wv    = nullptr;
static CUtensorMap* g_tma_Q     = nullptr;
static CUtensorMap* g_tma_K_cpu = nullptr;
static CUtensorMap* g_tma_V_cpu = nullptr;

static float* g_O_run   = nullptr;  static size_t g_O_run_cap   = 0;
static float* g_LSE_run = nullptr;  static size_t g_LSE_run_cap = 0;

static cudaError_t ensure_f32(float** buf, size_t* cap, size_t need) {
    if (need > *cap) {
        if (*buf) cudaFree(*buf);
        cudaError_t e = cudaMalloc(buf, need * sizeof(float));
        if (e != cudaSuccess) { *cap = 0; return e; }
        *cap = need;
    }
    return cudaSuccess;
}

// --------------------------------------------------------------------------
// Small kernel to fill a float buffer with a scalar value
// --------------------------------------------------------------------------
__global__ void fill_float_kernel(float* ptr, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = val;
}

// --------------------------------------------------------------------------
// TMA descriptor helpers (BF16, matching test_directkv_cpu_kv.cu)
// --------------------------------------------------------------------------

// K_cpu: [B, S_total, NH, D] row-major → 4D (D, NH, S, B), box={64,1,kBN,1}
static CUresult make_tma_K_cpu(CUtensorMap* desc, void const* ptr,
    int B, int S_total, int NH, int D, int blockN)
{
    uint64_t gd[4] = {(uint64_t)D, (uint64_t)NH, (uint64_t)S_total, (uint64_t)B};
    uint64_t gs[3] = {(uint64_t)D*2, (uint64_t)NH*D*2, (uint64_t)S_total*NH*D*2};
    uint32_t bx[4] = {64, 1, (uint32_t)blockN, 1};
    uint32_t es[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        const_cast<void*>(ptr), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// V_cpu: same global layout as K, but box={64,1,8,1} (MN-major V atom)
static CUresult make_tma_V_cpu(CUtensorMap* desc, void const* ptr,
    int B, int S_total, int NH, int D)
{
    uint64_t gd[4] = {(uint64_t)D, (uint64_t)NH, (uint64_t)S_total, (uint64_t)B};
    uint64_t gs[3] = {(uint64_t)D*2, (uint64_t)NH*D*2, (uint64_t)S_total*NH*D*2};
    uint32_t bx[4] = {64, 1, 8, 1};
    uint32_t es[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        const_cast<void*>(ptr), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// Q: [B, S_q, NH, D] → 4D (D, NH, S, B), box={64,1,64,1}
static CUresult make_tma_Q(CUtensorMap* desc, void const* ptr,
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

// X: [B, S_new, H] → 3D (H, S, B), box={kHC, kBN, 1}
static CUresult make_tma_X(CUtensorMap* desc, void const* ptr,
    int B, int S, int H, int hc, int blockN)
{
    uint64_t gd[3] = {(uint64_t)H, (uint64_t)S, (uint64_t)B};
    uint64_t gs[2] = {(uint64_t)H*2, (uint64_t)S*H*2};
    uint32_t bx[3] = {(uint32_t)hc, (uint32_t)blockN, 1};
    uint32_t es[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
        const_cast<void*>(ptr), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// W (Wk/Wv): [NH, D, H] → 3D (H, D, NH), box={kHC, D, 1}
static CUresult make_tma_W(CUtensorMap* desc, void const* ptr,
    int NH, int D, int H, int hc)
{
    uint64_t gd[3] = {(uint64_t)H, (uint64_t)D, (uint64_t)NH};
    uint64_t gs[2] = {(uint64_t)H*2, (uint64_t)D*H*2};
    uint32_t bx[3] = {(uint32_t)hc, (uint32_t)D, 1};
    uint32_t es[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
        const_cast<void*>(ptr), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// --------------------------------------------------------------------------
// Core forward implementation
// --------------------------------------------------------------------------
static torch::Tensor directkv_fwd_impl(
    torch::Tensor X,
    torch::Tensor Wk,
    torch::Tensor Wv,
    torch::Tensor Q,
    torch::Tensor K_cpu,
    torch::Tensor V_cpu,
    c10::optional<torch::Tensor> cos_sin,
    float softmax_scale,
    bool  is_causal,
    int   seqlen_past,
    int   q_pos_start)
{
    const int B        = (int)X.size(0);
    const int S_new    = (int)X.size(1);
    const int H        = (int)X.size(2);
    const int NH       = (int)Wk.size(0);    // KV heads (NH_kv)
    const int NH_q     = (int)Q.size(2);     // Q heads (NH_q; == NH for MHA)
    const int gqa_r    = NH_q / NH;          // GQA ratio (1 for MHA)
    const int S_total  = (int)K_cpu.size(1);

    TORCH_CHECK(S_total == seqlen_past + S_new,
        "K_cpu S_total (", S_total, ") != seqlen_past (", seqlen_past,
        ") + S_new (", S_new, ")");

    const int num_n_past   = seqlen_past / kBN;
    const int num_n_new    = S_new       / kBN;
    const int num_n_blocks = num_n_past + num_n_new;
    const int num_m_blocks = S_new / kBM;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // --- Persistent intermediate buffers (sized for NH_q, not NH) ---
    const size_t n_O   = (size_t)B * S_new * NH_q * kD;
    const size_t n_row = (size_t)B * S_new * NH_q;
    {
        cudaError_t e;
        if ((e = ensure_f32(&g_O_run,   &g_O_run_cap,   n_O))   != cudaSuccess)
            TORCH_CHECK(false, "cudaMalloc O_run: ", cudaGetErrorString(e));
        if ((e = ensure_f32(&g_LSE_run, &g_LSE_run_cap, n_row)) != cudaSuccess)
            TORCH_CHECK(false, "cudaMalloc LSE_run: ", cudaGetErrorString(e));
    }

    // Init O_run = 0.0
    TORCH_CHECK(cudaMemsetAsync(g_O_run, 0, n_O * sizeof(float), stream) == cudaSuccess,
        "cudaMemsetAsync O_run failed");

    // Init LSE_run = -FLT_MAX via a small fill kernel
    {
        int threads = 256;
        int blocks  = ((int)n_row + threads - 1) / threads;
        fill_float_kernel<<<blocks, threads, 0, stream>>>(
            g_LSE_run, -std::numeric_limits<float>::max(), (int)n_row);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fill_float_kernel failed");
    }

    // --- TMA descriptor device allocations (once) ---
    auto alloc_tma_desc = [](CUtensorMap** p) {
        if (!*p) {
            cudaError_t e = cudaMalloc(p, sizeof(CUtensorMap));
            TORCH_CHECK(e == cudaSuccess, "cudaMalloc TMA desc: ", cudaGetErrorString(e));
        }
    };
    alloc_tma_desc(&g_tma_X);
    alloc_tma_desc(&g_tma_Wk);
    alloc_tma_desc(&g_tma_Wv);
    alloc_tma_desc(&g_tma_Q);
    alloc_tma_desc(&g_tma_K_cpu);
    alloc_tma_desc(&g_tma_V_cpu);

    // --- Build host-side TMA descriptors ---
    auto cu_ok = [](CUresult r, const char* name) {
        TORCH_CHECK(r == CUDA_SUCCESS,
            "cuTensorMapEncodeTiled failed for ", name, ": err=", (int)r);
    };
    CUtensorMap h_X, h_Wk, h_Wv, h_Q, h_K, h_V;
    cu_ok(make_tma_X  (&h_X,  X.data_ptr(),      B, S_new,   H,   kHC, kBN), "X");
    cu_ok(make_tma_W  (&h_Wk, Wk.data_ptr(),     NH, kD,     H,   kHC),      "Wk");
    cu_ok(make_tma_W  (&h_Wv, Wv.data_ptr(),     NH, kD,     H,   kHC),      "Wv");
    cu_ok(make_tma_Q  (&h_Q,  Q.data_ptr(),      B, S_new,   NH_q, kD),      "Q"); // NH_q for GQA
    cu_ok(make_tma_K_cpu(&h_K, K_cpu.data_ptr(), B, S_total, NH,  kD, kBN), "K_cpu");
    cu_ok(make_tma_V_cpu(&h_V, V_cpu.data_ptr(), B, S_total, NH,  kD),      "V_cpu");

    // Copy to device (synchronous; descriptors are 128 bytes each)
    TORCH_CHECK(cudaMemcpy(g_tma_X,     &h_X,  sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(g_tma_Wk,    &h_Wk, sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(g_tma_Wv,    &h_Wv, sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(g_tma_Q,     &h_Q,  sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(g_tma_K_cpu, &h_K,  sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(g_tma_V_cpu, &h_V,  sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);

    // --- Output tensor: [B, S_new, NH_q, kD] ---
    auto O = torch::empty({B, S_new, NH_q, kD}, X.options().dtype(at::kBFloat16));

    // --- Build kernel params ---
    sm_parallel_v2_cpu_kv::SmParallelV2CpuKVParams p{};
    p.ptr_X       = X.data_ptr();
    p.ptr_Wk      = Wk.data_ptr();
    p.ptr_Wv      = Wv.data_ptr();
    p.ptr_Q       = Q.data_ptr();
    p.ptr_O       = O.data_ptr();
    p.ptr_O_run   = g_O_run;
    p.ptr_LSE_run = g_LSE_run;
    p.ptr_K_cpu   = K_cpu.data_ptr();
    p.ptr_V_cpu   = V_cpu.data_ptr();

    p.batch             = B;
    p.seqlen_new        = S_new;
    p.seqlen_q          = S_new;
    p.num_heads         = NH;    // KV heads (grid dim X)
    p.num_q_heads       = NH_q;  // Q heads (for O/LSE layout)
    p.gqa_ratio         = gqa_r; // Q-heads per KV-head (1 for MHA)
    p.head_dim          = kD;
    p.hidden_dim        = H;
    p.seqlen_past       = seqlen_past;
    p.num_n_blocks_past = num_n_past;
    p.num_n_blocks      = num_n_blocks;
    p.num_m_blocks      = num_m_blocks;
    p.softmax_scale     = softmax_scale;
    p.is_causal         = is_causal ? 1 : 0;

    // RoPE
    if (cos_sin.has_value() && cos_sin.value().defined()) {
        p.ptr_cos_sin = cos_sin.value().data_ptr<float>();
        p.rope_mode   = 1;   // neox_style
        p.rope_dim    = kD;
        p.q_pos_start = q_pos_start;
    } else {
        p.ptr_cos_sin = nullptr;
        p.rope_mode   = 0;
        p.rope_dim    = 0;
        p.q_pos_start = -1;
    }

    p.tma_X     = g_tma_X;
    p.tma_Wk    = g_tma_Wk;
    p.tma_Wv    = g_tma_Wv;
    p.tma_Q     = g_tma_Q;
    p.tma_K_cpu = g_tma_K_cpu;
    p.tma_V_cpu = g_tma_V_cpu;
    p.ptr_req_pool_indices = nullptr;   // non-graph path: K/V indexed by batch directly

    // --- Launch ---
    cudaError_t err =
        sm_parallel_v2_cpu_kv::launch_sm_parallel_v2_cpu_kv<Traits>(p, stream);
    TORCH_CHECK(err == cudaSuccess,
        "launch_sm_parallel_v2_cpu_kv failed: ", cudaGetErrorString(err));

    return O;
}

// --------------------------------------------------------------------------
// Per-layer TMA descriptor storage for CUDA graph (pre-built, stable addresses)
// --------------------------------------------------------------------------
#include <unordered_map>

struct LayerTmaSet {
    CUtensorMap* tma_X     = nullptr;
    CUtensorMap* tma_Wk    = nullptr;
    CUtensorMap* tma_Wv    = nullptr;
    CUtensorMap* tma_Q     = nullptr;
    CUtensorMap* tma_K_cpu = nullptr;
    CUtensorMap* tma_V_cpu = nullptr;
    // GPU int32 tensors holding num_n_blocks_past and num_n_blocks for replay
    int* ptr_num_n_blocks_past = nullptr;   // device memory
    int* ptr_num_n_blocks      = nullptr;   // device memory
    // GQA bookkeeping (set in directkv_init_tma, read in directkv_fwd_graph)
    int nh_kv        = 0;   // NH_kv (= Wk.size(0))
    int nh_q         = 0;   // NH_q  (= Q_max.size(2))
    int gqa_ratio    = 1;   // nh_q / nh_kv
};
static std::unordered_map<int, LayerTmaSet> g_cg_tma;

static void alloc_layer_tma(LayerTmaSet& s) {
    auto alloc = [](CUtensorMap** p) {
        if (!*p) {
            cudaError_t e = cudaMalloc(p, sizeof(CUtensorMap));
            TORCH_CHECK(e == cudaSuccess, "cudaMalloc tma: ", cudaGetErrorString(e));
        }
    };
    alloc(&s.tma_X);
    alloc(&s.tma_Wk);
    alloc(&s.tma_Wv);
    alloc(&s.tma_Q);
    alloc(&s.tma_K_cpu);
    alloc(&s.tma_V_cpu);
    if (!s.ptr_num_n_blocks_past) {
        TORCH_CHECK(cudaMalloc(&s.ptr_num_n_blocks_past, sizeof(int)) == cudaSuccess);
    }
    if (!s.ptr_num_n_blocks) {
        TORCH_CHECK(cudaMalloc(&s.ptr_num_n_blocks, sizeof(int)) == cudaSuccess);
    }
}

// --------------------------------------------------------------------------
// directkv_init_tma: pre-build all 6 TMA descriptors for one layer using
// max-shape tensors. Call once per layer during init_cuda_graph_state.
// Tensors must keep stable GPU/CPU addresses for the lifetime of the graph.
//
// X_max   : [max_bs, S_new_max, H]          GPU bf16  (S_new_max = 64 for decode)
// Wk, Wv  : [NH, D, H]                      GPU bf16  (layer weights, stable)
// Q_max   : [max_bs, S_new_max, NH, D]      GPU bf16
// K_max   : [max_bs, S_total_max, NH, D]    CPU-pinned bf16  (pool base slice)
// V_max   : [max_bs, S_total_max, NH, D]    CPU-pinned bf16
// --------------------------------------------------------------------------
void directkv_init_tma(
    int64_t layer_id,
    torch::Tensor X_max,
    torch::Tensor Wk,
    torch::Tensor Wv,
    torch::Tensor Q_max,
    torch::Tensor K_max,
    torch::Tensor V_max)
{
    SMPV2_CHECK_CUDA(X_max); SMPV2_CHECK_CUDA(Wk); SMPV2_CHECK_CUDA(Wv); SMPV2_CHECK_CUDA(Q_max);
    SMPV2_CHECK_CPU(K_max); SMPV2_CHECK_CPU(V_max);
    SMPV2_CHECK_CONTIG(X_max); SMPV2_CHECK_CONTIG(Wk); SMPV2_CHECK_CONTIG(Wv); SMPV2_CHECK_CONTIG(Q_max);
    SMPV2_CHECK_CONTIG(K_max); SMPV2_CHECK_CONTIG(V_max);
    SMPV2_CHECK_BF16(X_max); SMPV2_CHECK_BF16(Wk); SMPV2_CHECK_BF16(Wv); SMPV2_CHECK_BF16(Q_max);
    SMPV2_CHECK_BF16(K_max); SMPV2_CHECK_BF16(V_max);

    const int lid      = (int)layer_id;
    const int B        = (int)X_max.size(0);    // max_bs — used for X/Q TMA
    const int B_kv     = (int)K_max.size(0);    // n_pool — used for K/V TMA (pool rows)
    const int S_new    = (int)X_max.size(1);
    const int H        = (int)X_max.size(2);
    const int NH       = (int)Wk.size(0);       // KV heads
    const int NH_q     = (int)Q_max.size(2);    // Q heads (== NH for MHA)
    const int S_total  = (int)K_max.size(1);

    LayerTmaSet& s = g_cg_tma[lid];
    alloc_layer_tma(s);
    // Store GQA dims so directkv_fwd_graph can build params without Wk/Wv
    s.nh_kv     = NH;
    s.nh_q      = NH_q;
    s.gqa_ratio = NH_q / NH;

    auto cu_ok = [](CUresult r, const char* name) {
        TORCH_CHECK(r == CUDA_SUCCESS,
            "directkv_init_tma: cuTensorMapEncodeTiled failed for ", name, ": err=", (int)r);
    };

    CUtensorMap h_X, h_Wk, h_Wv, h_Q, h_K, h_V;
    cu_ok(make_tma_X    (&h_X,  X_max.data_ptr(),    B,    S_new,   H,   kHC, kBN), "X");
    cu_ok(make_tma_W    (&h_Wk, Wk.data_ptr(),       NH,   kD,      H,   kHC),      "Wk");
    cu_ok(make_tma_W    (&h_Wv, Wv.data_ptr(),       NH,   kD,      H,   kHC),      "Wv");
    cu_ok(make_tma_Q    (&h_Q,  Q_max.data_ptr(),    B,    S_new,   NH_q, kD),      "Q"); // NH_q for GQA
    cu_ok(make_tma_K_cpu(&h_K,  K_max.data_ptr(),    B_kv, S_total, NH,  kD, kBN), "K_cpu");
    cu_ok(make_tma_V_cpu(&h_V,  V_max.data_ptr(),    B_kv, S_total, NH,  kD),      "V_cpu");

    TORCH_CHECK(cudaMemcpy(s.tma_X,     &h_X,  sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(s.tma_Wk,    &h_Wk, sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(s.tma_Wv,    &h_Wv, sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(s.tma_Q,     &h_Q,  sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(s.tma_K_cpu, &h_K,  sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(s.tma_V_cpu, &h_V,  sizeof(CUtensorMap), cudaMemcpyHostToDevice) == cudaSuccess);

    // Pre-allocate O_run and LSE_run with max sizes to avoid cudaMalloc during
    // CUDA graph capture (cudaMalloc is illegal inside a captured stream).
    const size_t n_O_max   = (size_t)B * S_new * NH_q * kD;
    const size_t n_row_max = (size_t)B * S_new * NH_q;
    cudaError_t ea = ensure_f32(&g_O_run,   &g_O_run_cap,   n_O_max);
    cudaError_t eb = ensure_f32(&g_LSE_run, &g_LSE_run_cap, n_row_max);
    TORCH_CHECK(ea == cudaSuccess, "directkv_init_tma: pre-alloc O_run failed: ", cudaGetErrorString(ea));
    TORCH_CHECK(eb == cudaSuccess, "directkv_init_tma: pre-alloc LSE_run failed: ", cudaGetErrorString(eb));
}

// --------------------------------------------------------------------------
// directkv_fwd_graph: graph-safe forward (no TMA rebuild, reads num_n_blocks
// from device memory). Requires directkv_init_tma to have been called first.
//
// X_pad   : [B, S_new, H]            GPU bf16  (zeroed outside slot, stable ptr)
// Q_pad   : [B, S_new, NH, D]        GPU bf16  (same)
// O_out   : [B, S_new, NH, D]        GPU bf16  (output, pre-allocated)
// n_past_dev: [1] int32 GPU — device tensor holding num_n_blocks_past
// n_blks_dev: [1] int32 GPU — device tensor holding num_n_blocks
// All shapes must match those used in directkv_init_tma for this layer.
// --------------------------------------------------------------------------
torch::Tensor directkv_fwd_graph(
    int64_t layer_id,
    torch::Tensor X_pad,
    torch::Tensor Q_pad,
    torch::Tensor O_out,
    torch::Tensor n_past_dev,
    torch::Tensor n_blks_dev,
    c10::optional<torch::Tensor> cos_sin,
    double softmax_scale,
    bool   is_causal,
    int64_t S_total_max,
    int64_t seqlen_past_max,
    c10::optional<torch::Tensor> req_pool_indices)
{
    SMPV2_CHECK_CUDA(X_pad); SMPV2_CHECK_CUDA(Q_pad); SMPV2_CHECK_CUDA(O_out);
    SMPV2_CHECK_CUDA(n_past_dev); SMPV2_CHECK_CUDA(n_blks_dev);
    SMPV2_CHECK_CONTIG(X_pad); SMPV2_CHECK_CONTIG(Q_pad); SMPV2_CHECK_CONTIG(O_out);
    SMPV2_CHECK_BF16(X_pad); SMPV2_CHECK_BF16(Q_pad); SMPV2_CHECK_BF16(O_out);
    TORCH_CHECK(n_past_dev.scalar_type() == at::kInt, "n_past_dev must be int32");
    TORCH_CHECK(n_blks_dev.scalar_type() == at::kInt, "n_blks_dev must be int32");

    const int lid   = (int)layer_id;
    TORCH_CHECK(g_cg_tma.count(lid) > 0,
        "directkv_fwd_graph: layer ", lid, " not initialized — call directkv_init_tma first");

    LayerTmaSet& s = g_cg_tma[lid];

    const int B     = (int)X_pad.size(0);
    const int S_new = (int)X_pad.size(1);
    const int H     = (int)X_pad.size(2);
    // Read NH_kv and NH_q from the TmaSet stored during directkv_init_tma.
    // Q_pad.size(2) == NH_q (backend allocates cg_Q_pad with NH_q heads).
    const int NH_kv = s.nh_kv;   // KV heads (grid dim)
    const int NH_q  = s.nh_q;    // Q heads (O/LSE layout)

    // Copy current step's num_n_blocks_past / num_n_blocks into the
    // device ints owned by this layer's TmaSet (captured as device-to-device copies).
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(cudaMemcpyAsync(s.ptr_num_n_blocks_past, n_past_dev.data_ptr<int>(),
        sizeof(int), cudaMemcpyDeviceToDevice, stream) == cudaSuccess);
    TORCH_CHECK(cudaMemcpyAsync(s.ptr_num_n_blocks, n_blks_dev.data_ptr<int>(),
        sizeof(int), cudaMemcpyDeviceToDevice, stream) == cudaSuccess);

    // Ensure O/LSE buffers are large enough (sized by NH_q for GQA)
    const size_t n_O   = (size_t)B * S_new * NH_q * kD;
    const size_t n_row = (size_t)B * S_new * NH_q;
    {
        cudaError_t e;
        if ((e = ensure_f32(&g_O_run,   &g_O_run_cap,   n_O))   != cudaSuccess)
            TORCH_CHECK(false, "cudaMalloc O_run: ", cudaGetErrorString(e));
        if ((e = ensure_f32(&g_LSE_run, &g_LSE_run_cap, n_row)) != cudaSuccess)
            TORCH_CHECK(false, "cudaMalloc LSE_run: ", cudaGetErrorString(e));
    }
    TORCH_CHECK(cudaMemsetAsync(g_O_run, 0, n_O * sizeof(float), stream) == cudaSuccess);
    {
        int threads = 256;
        int blocks  = ((int)n_row + threads - 1) / threads;
        fill_float_kernel<<<blocks, threads, 0, stream>>>(
            g_LSE_run, -std::numeric_limits<float>::max(), (int)n_row);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fill_float_kernel failed");
    }

    // Use the max num_n_blocks captured in s for grid/loop bounds at capture time.
    // ptr_num_n_blocks_past and ptr_num_n_blocks will be read from device memory
    // at kernel execution time (updated above via async D2D copy).
    const int num_n_new    = S_new / kBN;
    const int num_n_blocks_max = (int)(seqlen_past_max / kBN) + num_n_new;
    const int num_m_blocks = S_new / kBM;

    sm_parallel_v2_cpu_kv::SmParallelV2CpuKVParams p{};
    p.ptr_X       = X_pad.data_ptr();
    p.ptr_Wk      = nullptr;   // not needed — projection uses TMA Wk
    p.ptr_Wv      = nullptr;
    p.ptr_Q       = Q_pad.data_ptr();
    p.ptr_O       = O_out.data_ptr();
    p.ptr_O_run   = g_O_run;
    p.ptr_LSE_run = g_LSE_run;
    p.ptr_K_cpu   = nullptr;   // not needed — TMA K_cpu has stable address
    p.ptr_V_cpu   = nullptr;

    p.batch             = B;
    p.seqlen_new        = S_new;
    p.seqlen_q          = S_new;
    p.num_heads         = NH_kv;         // KV heads (grid dim X)
    p.num_q_heads       = NH_q;          // Q heads (O/LSE layout)
    p.gqa_ratio         = s.gqa_ratio;   // Q-heads per KV-head
    p.head_dim          = kD;
    p.hidden_dim        = H;
    p.seqlen_past       = (int)seqlen_past_max;
    p.num_n_blocks_past = (int)(seqlen_past_max / kBN);
    p.num_n_blocks      = num_n_blocks_max;
    p.num_m_blocks      = num_m_blocks;
    p.softmax_scale     = (float)softmax_scale;
    p.is_causal         = is_causal ? 1 : 0;

    // Dynamic override via device pointers (kernel reads these at runtime)
    p.ptr_num_n_blocks_past = s.ptr_num_n_blocks_past;
    p.ptr_num_n_blocks      = s.ptr_num_n_blocks;

    if (cos_sin.has_value() && cos_sin.value().defined()) {
        p.ptr_cos_sin = cos_sin.value().data_ptr<float>();
        p.rope_mode   = 1;
        p.rope_dim    = kD;
        p.q_pos_start = -1;
    } else {
        p.ptr_cos_sin = nullptr;
        p.rope_mode   = 0;
        p.rope_dim    = 0;
        p.q_pos_start = -1;
    }

    p.tma_X     = s.tma_X;
    p.tma_Wk    = s.tma_Wk;
    p.tma_Wv    = s.tma_Wv;
    p.tma_Q     = s.tma_Q;
    p.tma_K_cpu = s.tma_K_cpu;
    p.tma_V_cpu = s.tma_V_cpu;

    if (req_pool_indices.has_value() && req_pool_indices.value().defined()) {
        SMPV2_CHECK_CUDA(req_pool_indices.value());
        TORCH_CHECK(req_pool_indices.value().scalar_type() == at::kInt,
            "req_pool_indices must be int32");
        p.ptr_req_pool_indices = req_pool_indices.value().data_ptr<int>();
    } else {
        p.ptr_req_pool_indices = nullptr;
    }

    cudaError_t err =
        sm_parallel_v2_cpu_kv::launch_sm_parallel_v2_cpu_kv<Traits>(p, stream);
    TORCH_CHECK(err == cudaSuccess,
        "directkv_fwd_graph: launch failed: ", cudaGetErrorString(err));

    return O_out;
}

// --------------------------------------------------------------------------
// Python-visible entry point
// --------------------------------------------------------------------------
torch::Tensor directkv_fwd(
    torch::Tensor X,
    torch::Tensor Wk,
    torch::Tensor Wv,
    torch::Tensor Q,
    torch::Tensor K_cpu,
    torch::Tensor V_cpu,
    c10::optional<torch::Tensor> cos_sin,
    double  softmax_scale,
    bool    is_causal,
    int64_t seqlen_past,
    int64_t q_pos_start)
{
    // Shape checks
    TORCH_CHECK(X.dim()     == 3, "X must be 3D [B, S_new, H], got ", X.dim(), "D");
    TORCH_CHECK(Wk.dim()    == 3, "Wk must be 3D [NH, D, H]");
    TORCH_CHECK(Wv.dim()    == 3, "Wv must be 3D [NH, D, H]");
    TORCH_CHECK(Q.dim()     == 4, "Q must be 4D [B, S_new, NH, D]");
    TORCH_CHECK(K_cpu.dim() == 4, "K_cpu must be 4D [B, S_total, NH, D]");
    TORCH_CHECK(V_cpu.dim() == 4, "V_cpu must be 4D [B, S_total, NH, D]");

    // Device checks
    SMPV2_CHECK_CUDA(X); SMPV2_CHECK_CUDA(Wk); SMPV2_CHECK_CUDA(Wv); SMPV2_CHECK_CUDA(Q);
    SMPV2_CHECK_CPU(K_cpu); SMPV2_CHECK_CPU(V_cpu);

    // Contiguity
    SMPV2_CHECK_CONTIG(X); SMPV2_CHECK_CONTIG(Wk); SMPV2_CHECK_CONTIG(Wv); SMPV2_CHECK_CONTIG(Q);
    SMPV2_CHECK_CONTIG(K_cpu); SMPV2_CHECK_CONTIG(V_cpu);

    // Dtype
    SMPV2_CHECK_BF16(X); SMPV2_CHECK_BF16(Wk); SMPV2_CHECK_BF16(Wv); SMPV2_CHECK_BF16(Q);
    SMPV2_CHECK_BF16(K_cpu); SMPV2_CHECK_BF16(V_cpu);

    // head_dim constraint
    TORCH_CHECK(Q.size(3) == kD,
        "head_dim must be ", kD, " (DefaultTraits_hdim128_bf16), got ", Q.size(3));

    // GQA constraint: NH_q must be a multiple of NH_kv
    TORCH_CHECK(Q.size(2) % Wk.size(0) == 0,
        "Q.size(2) (NH_q=", Q.size(2), ") must be a multiple of "
        "Wk.size(0) (NH_kv=", Wk.size(0), ")");

    // Alignment constraints
    TORCH_CHECK(seqlen_past % kBN == 0,
        "seqlen_past (", seqlen_past, ") must be a multiple of kBN=", kBN);
    TORCH_CHECK(X.size(1) % kBN == 0,
        "S_new (", X.size(1), ") must be a multiple of kBN=", kBN);

    // cos_sin validation
    if (cos_sin.has_value() && cos_sin.value().defined()) {
        SMPV2_CHECK_CUDA(cos_sin.value());
        SMPV2_CHECK_F32(cos_sin.value());
        TORCH_CHECK(cos_sin.value().dim() == 2,
            "cos_sin must be 2D [max_pos, head_dim]");
        TORCH_CHECK(cos_sin.value().size(1) == kD,
            "cos_sin.size(1) must equal head_dim=", kD,
            ", got ", cos_sin.value().size(1));
    }

    return directkv_fwd_impl(
        X, Wk, Wv, Q, K_cpu, V_cpu, cos_sin,
        (float)softmax_scale, is_causal,
        (int)seqlen_past, (int)q_pos_start);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DirectKV CPU-KV: fused X->KV projection + Neox RoPE + attention (SM90 bf16)";
    m.def("directkv_fwd", &directkv_fwd,
        py::arg("X"),
        py::arg("Wk"),
        py::arg("Wv"),
        py::arg("Q"),
        py::arg("K_cpu"),
        py::arg("V_cpu"),
        py::arg("cos_sin"),
        py::arg("softmax_scale"),
        py::arg("is_causal"),
        py::arg("seqlen_past"),
        py::arg("q_pos_start"),
        "DirectKV forward pass.\n\n"
        "Args:\n"
        "  X        (bf16 GPU [B,S_new,H]):       input activations\n"
        "  Wk       (bf16 GPU [NH,D,H]):           K projection weights\n"
        "  Wv       (bf16 GPU [NH,D,H]):           V projection weights\n"
        "  Q        (bf16 GPU [B,S_new,NH,D]):     query (pre-projected)\n"
        "  K_cpu    (bf16 CPU-pinned [B,S_tot,NH,D]): KV cache K\n"
        "  V_cpu    (bf16 CPU-pinned [B,S_tot,NH,D]): KV cache V\n"
        "  cos_sin  (fp32 GPU [max_pos,D] | None): RoPE cos/sin table\n"
        "  softmax_scale (float): 1/sqrt(D)\n"
        "  is_causal (bool): apply causal mask\n"
        "  seqlen_past (int): number of past KV tokens (multiple of 64)\n"
        "  q_pos_start (int): global position index of first Q token\n"
        "Returns:\n"
        "  O (bf16 GPU [B,S_new,NH,D]): attention output");

    m.def("directkv_init_tma", &directkv_init_tma,
        py::arg("layer_id"),
        py::arg("X_max"),
        py::arg("Wk"),
        py::arg("Wv"),
        py::arg("Q_max"),
        py::arg("K_max"),
        py::arg("V_max"),
        "Pre-build TMA descriptors for one layer (CUDA graph init).\n"
        "Must be called once per layer before directkv_fwd_graph.\n"
        "All tensor shapes must be the max shapes that will be used during graph replay.");

    m.def("directkv_fwd_graph", &directkv_fwd_graph,
        py::arg("layer_id"),
        py::arg("X_pad"),
        py::arg("Q_pad"),
        py::arg("O_out"),
        py::arg("n_past_dev"),
        py::arg("n_blks_dev"),
        py::arg("cos_sin"),
        py::arg("softmax_scale"),
        py::arg("is_causal"),
        py::arg("S_total_max"),
        py::arg("seqlen_past_max"),
        py::arg("req_pool_indices"),
        "Graph-safe DirectKV forward (no TMA rebuild).\n"
        "Reads num_n_blocks_past / num_n_blocks from device memory tensors.\n"
        "req_pool_indices: int32 GPU tensor [bs] mapping batch position to pool row.\n"
        "directkv_init_tma must have been called first for this layer_id.\n"
        "Returns O_out (same tensor, modified in-place and returned for chaining).");
}
