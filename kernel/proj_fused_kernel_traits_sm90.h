/******************************************************************************
 * Projection-Fused Flash Attention — Kernel Traits (SM90)
 *
 * Phase-2: traits now include CuTe WGMMA tiled-MMA objects and swizzled
 * smem layouts compatible with Hopper GMMA. The kernel uses a single
 * warpgroup (128 MMA threads) + 32 storer threads = 160 threads/block.
 ******************************************************************************/
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>

namespace proj_fused {

template <
    int kBlockM_,
    int kBlockN_,
    int kHeadDim_,
    int kHiddenDim_,
    int kHiddenChunk_,
    class Element_ = cutlass::half_t>
struct ProjFusedKernelTraitsSm90 {
    using Element = Element_;

    static constexpr int kBlockM      = kBlockM_;
    static constexpr int kBlockN      = kBlockN_;
    static constexpr int kHeadDim     = kHeadDim_;
    static constexpr int kHiddenDim   = kHiddenDim_;
    static constexpr int kHiddenChunk = kHiddenChunk_;
    static_assert(kHiddenDim % kHiddenChunk == 0, "chunk divides hidden");
    static constexpr int kNumProjChunks = kHiddenDim / kHiddenChunk;

    static constexpr int kNWarpsMMA   = 4;
    static constexpr int kNThreadsMMA = kNWarpsMMA * 32;    // 128
    static constexpr int kNThreadsStorer = 32;
    static constexpr int kNThreads    = kNThreadsMMA + kNThreadsStorer;  // 160

    // -------------------- Tiled MMAs (WGMMA) --------------------
    // Projection GEMM tile: always 64 rows (1 warpgroup M-tile).
    // For kBlockN > 64, kernel loops over kBlockN/64 halves.
    static constexpr int kProjTileM = 64;
    static constexpr int kNumProjHalves = kBlockN / kProjTileM;
    using TileShape_Proj = cute::Shape<cute::Int<kProjTileM>, cute::Int<kHeadDim>, cute::Int<kHiddenChunk>>;
    using TiledMmaProj = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, float, TileShape_Proj>(),
        cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>{}));

    // Attention GEMM-I: S[kBlockM,kBlockN] = Q[kBlockM,kHeadDim] @ K^T[kHeadDim,kBlockN]
    using TileShape_QK = cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockN>, cute::Int<kHeadDim>>;
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, float, TileShape_QK>(),
        cute::Layout<cute::Shape<cute::Int<kBlockM/64>, cute::_1, cute::_1>>{}));

    // Attention GEMM-II: O[kBlockM,kHeadDim] = P[kBlockM,kBlockN] @ V[kBlockN,kHeadDim]
    // RS-mode: P in regs, V in smem (MN-major).
    using TileShape_PV = cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>, cute::Int<kBlockN>>;
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, float, TileShape_PV,
                                   cute::GMMA::Major::K, cute::GMMA::Major::MN>(),
        cute::Layout<cute::Shape<cute::Int<kBlockM/64>, cute::_1, cute::_1>>{}));

    // -------------------- SMEM layouts (swizzled for WGMMA) --------------------
    // P[kBlockM, kBlockN] K-major swizzled (for SS-mode PV)
    using SmemLayoutAtomP = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element, cute::Int<kBlockM>, cute::Int<kBlockN>>());
    using SmemLayoutP = decltype(cute::tile_to_shape(
        SmemLayoutAtomP{}, cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kBlockN>{})));

    using SmemLayoutAtomQ = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element, cute::Int<kBlockM>, cute::Int<kHeadDim>>());
    using SmemLayoutQ = decltype(cute::tile_to_shape(
        SmemLayoutAtomQ{}, cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kHeadDim>{})));

    using SmemLayoutAtomK = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element, cute::Int<kBlockN>, cute::Int<kHeadDim>>());
    using SmemLayoutK = decltype(cute::tile_to_shape(
        SmemLayoutAtomK{}, cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kHeadDim>{})));

    // V is MN-major (transposed in smem) so PV GEMM can use V directly.
    // For the PV GEMM, the B operand has shape (N=kHeadDim, K=kBlockN). So sV is
    // stored as V^T, shape (kHeadDim, kBlockN), MN-major (contiguous on kHeadDim).
    using SmemLayoutAtomV = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::MN, Element, cute::Int<kHeadDim>, cute::Int<kBlockN>>());
    using SmemLayoutV = decltype(cute::tile_to_shape(
        SmemLayoutAtomV{}, cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<kBlockN>{})));

    // sX uses kProjTileM rows (always 64), not kBlockN, so it fits in sQ0 alias
    using SmemLayoutAtomX = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element, cute::Int<kProjTileM>, cute::Int<kHiddenChunk>>());
    using SmemLayoutX = decltype(cute::tile_to_shape(
        SmemLayoutAtomX{}, cute::make_shape(cute::Int<kProjTileM>{}, cute::Int<kHiddenChunk>{})));

    // Wk/Wv: shape [kHeadDim, kHiddenChunk] in gmem ([D,H] row-major); K-major in smem.
    using SmemLayoutAtomW = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element, cute::Int<kHeadDim>, cute::Int<kHiddenChunk>>());
    using SmemLayoutW = decltype(cute::tile_to_shape(
        SmemLayoutAtomW{}, cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<kHiddenChunk>{})));

    // O[kBlockM, kHeadDim] K-major swizzled (for STSM output staging)
    using SmemLayoutAtomO = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            cute::GMMA::Major::K, Element, cute::Int<kBlockM>, cute::Int<kHeadDim>>());
    using SmemLayoutO = decltype(cute::tile_to_shape(
        SmemLayoutAtomO{}, cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kHeadDim>{})));

    // Byte sizes
    static constexpr int kSmemBytesQ = cute::cosize(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemBytesK = cute::cosize(SmemLayoutK{}) * sizeof(Element);
    static constexpr int kSmemBytesV = cute::cosize(SmemLayoutV{}) * sizeof(Element);
    static constexpr int kSmemBytesX = cute::cosize(SmemLayoutX{}) * sizeof(Element);
    static constexpr int kSmemBytesW = cute::cosize(SmemLayoutW{}) * sizeof(Element);
    static constexpr int kSmemBytesP = cute::cosize(SmemLayoutP{}) * sizeof(Element);
    static constexpr int kSmemBytesO = cute::cosize(SmemLayoutO{}) * sizeof(Element);
};

using DefaultTraits_hdim128_fp16 = ProjFusedKernelTraitsSm90<
    /*kBlockM=*/64, /*kBlockN=*/64, /*kHeadDim=*/128,
    /*kHiddenDim=*/128, /*kHiddenChunk=*/64, cutlass::half_t>;

// Tile-size exploration variants. Architectural constraints with the current
// single-warpgroup (128 MMA threads) kernel:
//   - kBlockM ≤ 64: AtomLayoutQK / AtomLayoutPV = <kBlockM/64,1,1>; larger would
//     need 2 warpgroups (256 MMA threads) for the QK/PV GEMMs.
//   - kBlockN ≤ 64: TiledMmaProj M-dim is kBlockN (projecting a tile of kBlockN
//     rows), with AtomLayoutProj = <kBlockN/64,1,1>; kBlockN=128 would also need
//     2 warpgroups for the projection GEMM.
// So the only dial available without a 2-warpgroup rewrite is kHiddenChunk.
using Traits_M64_N64_HC128_hdim128_fp16 = ProjFusedKernelTraitsSm90<
    /*kBlockM=*/64, /*kBlockN=*/64, /*kHeadDim=*/128,
    /*kHiddenDim=*/128, /*kHiddenChunk=*/128, cutlass::half_t>;

using DefaultTraits_hdim64_fp16 = ProjFusedKernelTraitsSm90<
    /*kBlockM=*/64, /*kBlockN=*/64, /*kHeadDim=*/64,
    /*kHiddenDim=*/64, /*kHiddenChunk=*/64, cutlass::half_t>;

using DefaultTraits_hdim128_bf16 = ProjFusedKernelTraitsSm90<
    /*kBlockM=*/64, /*kBlockN=*/64, /*kHeadDim=*/128,
    /*kHiddenDim=*/128, /*kHiddenChunk=*/64, cutlass::bfloat16_t>;

} // namespace proj_fused
