#pragma once
#include <cuda_bf16.h>
#include <cutlass/arch/wmma.h>

// CUTLASS da5e086 has no BF16 Wmma operator (and maps BF16 types only in the device pass).
// This is the m16n16k16 BF16 mma_sync with FP32 accumulation that cuBLAS's wmma kernels use.
namespace cutlass {
namespace arch {
#if !defined(__CUDA_ARCH__)
template <>
struct CutlassToWmmaDataType<bfloat16_t> {
  using Type = __nv_bfloat16;
};
template <>
struct WmmaToCutlassDataType<__nv_bfloat16> {
  using Type = bfloat16_t;
};
#endif
template <>
struct Wmma<gemm::GemmShape<16, 16, 16>, bfloat16_t, layout::RowMajor, bfloat16_t, layout::ColumnMajor, float,
            layout::RowMajor, OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 16>;
  using ElementA = bfloat16_t;
  using ElementB = bfloat16_t;
  using ElementC = float;
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::ColumnMajor;
  using LayoutC = layout::RowMajor;
  using Operator = OpMultiplyAdd;
  using ArchTag = Sm80;
  using FragmentA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major>;
  using FragmentB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major>;
  using FragmentC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;
  CUTLASS_DEVICE void operator()(FragmentC& d, FragmentA const& a, FragmentB const& b, FragmentC const& c) const {
    nvcuda::wmma::mma_sync(d, a, b, c);
  }
};
}  // namespace arch
}  // namespace cutlass
