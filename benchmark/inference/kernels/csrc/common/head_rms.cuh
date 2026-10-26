// D-wide projection with SDM's unweighted per-head RMS fused into the epilogue. SM120, BF16.
//
//   out = rms(bf16(a @ w.T + bias)) per head of Head channels, eps 1e-6, no weight
//
// a [M, D] (lda D), w [D, D] row-major (the leading D rows of a larger weight are fine), bias [D],
// out [M, D] with row stride ld_out. One N tile spans all D columns, so a CTA has read all of its A rows
// before it stores: out may be exactly a. Without the RMS (Rms = false) this is the plain addmm.
// Bytes match cuBLAS's addmm (CUTLASS 128x256x32 tile, no split-K) followed by SDM's Triton rmsnorm_cast
// at channels = Head (32 or 64). Site is an empty tag type that only names the caller in the kernel symbol.
#pragma once
#include <climits>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include "common/rms_ptx.cuh"

namespace head_rms {
using BF = cutlass::bfloat16_t;
using Bias = cutlass::epilogue::thread::LinearCombination<BF, 8, float, float>;
using rms_ptx::add;
using rms_ptx::inv_rms;
using rms_ptx::mul;
using rms_ptx::square_add;

// Each lane holds 8 adjacent channels (two rmsnorm_cast groups of 4) and Head / 8 adjacent lanes hold one head.
// rmsnorm_cast squares each group of 4 in channel order, then reduces the Head / 4 group sums by an XOR
// butterfly with group shifts Head/8, ..., 2, 1. Lane shifts Head/16, ..., 1 are group shifts
// Head/8, ..., 2; the in-lane add of the two groups comes last as group shift 1.
template <class Base, int Head, class Site>
struct HeadRmsEpilogue : Base {
  using Iterator = typename Base::OutputTileIterator;
  using Fragment = typename Iterator::Fragment;
  using Map = typename Iterator::ThreadMap;
  static constexpr int kLanes = Head / 8;
  static_assert(Iterator::kElementsPerAccess == 8 && Head % 16 == 0 && kLanes <= 32, "8 channels per lane");
  static_assert(Map::Detail::kAccessWidth % kLanes == 0 && Map::Delta::kColumn % Head == 0,
                "aligned groups of Head/8 lanes each cover one head of one row");
  using Base::Base;

  struct Source : Base::SourceAspectNeeded {
    using Base::SourceAspectNeeded::SourceAspectNeeded;
    CUTLASS_DEVICE void apply_output_operator(Fragment& out, Bias const& op,
                                              typename Base::SharedLoadIterator::Fragment const& acc) {
      Base::SourceAspectNeeded::apply_output_operator(out, op, acc, this->source_fragment);
      auto* v = reinterpret_cast<cutlass::Array<BF, 8>*>(&out);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Fragment::kElements / 8; ++j) {
        float s0 = 0.f, s1 = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          s0 = square_add(BF(v[j][i]).raw(), s0);
          s1 = square_add(BF(v[j][i + 4]).raw(), s1);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int shift = kLanes / 2; shift >= 1; shift >>= 1) {
          s0 = add(s0, __shfl_xor_sync(0xffffffffu, s0, shift, kLanes));
          s1 = add(s1, __shfl_xor_sync(0xffffffffu, s1, shift, kLanes));
        }
        const float scale = inv_rms<Head>(add(s0, s1));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) v[j][i] = BF(mul(float(v[j][i]), scale));
      }
    }
  };

  CUTLASS_DEVICE void operator()(Bias const& op, Iterator destination, typename Base::AccumulatorTile const& acc,
                                 Iterator bias) {
    Base::operator()(op, destination, acc, Source(bias));
  }
};

template <int D>
using Gemm = cutlass::gemm::device::Gemm<
    BF, cutlass::layout::RowMajor, BF, cutlass::layout::ColumnMajor, BF, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, D, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, Bias,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 3, 8, 8, false, cutlass::arch::OpMultiplyAdd>;

template <int D, int Head, bool Rms, class Site>
using Kernel = cutlass::gemm::kernel::Gemm<
    typename Gemm<D>::GemmKernel::Mma,
    std::conditional_t<Rms, HeadRmsEpilogue<typename Gemm<D>::GemmKernel::Epilogue, Head, Site>,
                       typename Gemm<D>::GemmKernel::Epilogue>,
    typename Gemm<D>::GemmKernel::ThreadblockSwizzle, false>;

constexpr int64_t kMaxM = INT_MAX - 128;  // the swizzle computes M + tile_m in int32

template <int D, int Head, bool Rms, class Site>
void launch(BF const* a, BF const* w, BF const* bias, BF* out, int ld_out, int m, cudaStream_t stream) {
  using G = Gemm<D>;
  using K = Kernel<D, Head, Rms, Site>;
  static_assert(sizeof(typename K::SharedStorage) == sizeof(typename G::GemmKernel::SharedStorage), "no extra smem");
  typename G::Arguments args({m, D, D}, {a, D}, {w, D}, {bias, 0}, {out, ld_out}, {1.f, 1.f}, 1);
  TORCH_CHECK(G::can_implement(args) == cutlass::Status::kSuccess, "head_rms: CUTLASS cannot implement this GEMM");
  typename G::ThreadblockSwizzle swizzle;
  auto tiled = swizzle.get_tiled_shape(args.problem_size, {128, D, 32}, 1);
  typename K::Params params(args.problem_size, tiled, args.ref_A.non_const_ref(), args.ref_B.non_const_ref(),
                            args.ref_C.non_const_ref(), args.ref_D, args.epilogue, nullptr);
  constexpr int smem = sizeof(typename K::SharedStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cutlass::Kernel<K><<<swizzle.get_grid_shape(tiled), K::kThreadCount, smem, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace head_rms
