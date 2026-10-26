// Column attention out-projection + residual + weighted RMSNorm + BF16 cast in one CUTLASS GEMM
// (SM120, BF16, D in {128, 256}). Rows r = f*C + c of [F, C, D]:
//
//   raw[r]  = RN_bf16(a[r] @ w.T + bias)          cuBLAS's addmm: 128 x D x 32 CTA tile (one N tile per row),
//                                                 64x64x32 warps, mma 16x8x16, 3 stages, no split-K
//   x[r]    = RN_bf16(raw[r] + table[c, P + f])   one FP32 add
//   norm[r] = RN_bf16(RN(x[r] * inv) * weight)    inv = rsqrt.approx.ftz(div.full(sum_c x_c^2, D) + eps)
//
// The GEMM destination receives raw, which stays live for the column-down residual.
//
// Reduction order is SDM's Triton rmsnorm_cast: group g = channels 4g..4g+3, fma from 0 in channel order,
// butterfly 16,8,4,2,1 over each run of 32 groups (one 128-channel half), halves added last. In the
// epilogue 16 lanes own a row, lane l holds channels 8l..8l+7 of each half: groups 2l (s0) and 2l+1 (s1).
// Lane XOR 8,4,2,1 is group XOR 16,8,4,2 and add(s0, s1) is group XOR 1.
#include <climits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include "common/rms_ptx.cuh"
#include "common/row_source.cuh"

namespace projection_rms {
using BF = cutlass::bfloat16_t;
using Vector = cutlass::AlignedArray<BF, 8, 16>;
using BiasOp = cutlass::epilogue::thread::LinearCombination<BF, 8, float, float>;

struct Op : BiasOp {
  struct Params : BiasOp::Params {
    BF const* table = nullptr;
    float const* weight = nullptr;
    BF* norm = nullptr;
    int64_t rows = 0, contexts = 1, row_features = 0, prefix = 0;
    float eps = 0.f;
    CUTLASS_HOST_DEVICE Params() : BiasOp::Params(1.f, 1.f) {}
  };
  Params parameters;
  CUTLASS_HOST_DEVICE Op(Params const& p) : BiasOp(p), parameters(p) {}
};

using rms_ptx::add;
using rms_ptx::mul;
using rms_ptx::square_add;
CUTLASS_DEVICE float inverse(float sum, float divisor, float eps) {
  float v;
  asm volatile("div.full.f32 %0,%1,%2;" : "=f"(v) : "f"(sum), "f"(divisor));
  v = add(v, eps);
  asm volatile("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(v) : "f"(v));
  return v;
}
// Squares of one 128-channel half; result is identical in all 16 lanes of the row.
CUTLASS_DEVICE float half_sum(Vector const& z) {
  float s0 = 0.f, s1 = 0.f;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 4; ++i) {
    s0 = square_add(BF(z[i]).raw(), s0);
    s1 = square_add(BF(z[i + 4]).raw(), s1);
  }
  CUTLASS_PRAGMA_UNROLL
  for (int shift = 8; shift >= 1; shift >>= 1) {
    s0 = add(s0, __shfl_xor_sync(0xffffffffu, s0, shift, 16));
    s1 = add(s1, __shfl_xor_sync(0xffffffffu, s1, shift, 16));
  }
  return add(s0, s1);
}

// Named for the kernel symbol profiles match on.
template <int D, class Base>
struct ColumnProjectionRms : Base {
  using Iterator = typename Base::OutputTileIterator;
  using Fragment = typename Iterator::Fragment;
  using SharedFragment = typename Base::SharedLoadIterator::Fragment;
  using Map = typename Iterator::ThreadMap;
  static constexpr int kHalves = D / 128;
  static_assert(Iterator::kElementsPerAccess == 8 && Fragment::kElements == SharedFragment::kElements,
                "8-channel vectors");
  static_assert(Map::Detail::kAccessWidth == 16 && Map::Iterations::kColumn == kHalves &&
                    (kHalves == 1 || Map::Delta::kColumn == 128),
                "16 lanes per row, one 128-channel half per vector");

  CUTLASS_DEVICE ColumnProjectionRms(typename Base::SharedStorage& s, int t, int w, int l) : Base(s, t, w, l) {}

  struct Source : row_source::BiasSource<Iterator> {
    CUTLASS_DEVICE void apply_output_operator(Fragment& out, Op const& op, SharedFragment const& acc) {
      auto const* av = reinterpret_cast<typename BiasOp::FragmentAccumulator const*>(&acc);
      auto const* bv = reinterpret_cast<typename BiasOp::FragmentSource const*>(&this->bias);
      auto* ov = reinterpret_cast<typename BiasOp::FragmentOutput*>(&out);
      auto const& p = op.parameters;
      const int start_column = this->start.column();
      row_source::for_each_row<Map>(this->start, [&](int64_t row, int fr) {
        const bool valid = row < p.rows;
        const int64_t f = row / p.contexts, c = row - f * p.contexts;
        const int64_t residual = (c * p.row_features + p.prefix + f) * D;
        Vector x[kHalves];
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < kHalves; ++h) {
          const int idx = fr * kHalves + h, col = start_column + h * 128;
          const auto raw = static_cast<BiasOp const&>(op)(av[idx], bv[idx]);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 8; ++i) x[h][i] = BF(0.f);
          if (valid) {
            Vector t;
            cutlass::arch::global_load<Vector, 16>(t, p.table + residual + col, true);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 8; ++i) x[h][i] = BF(add(float(raw[i]), float(t[i])));
          }
          ov[idx] = raw;
        }
        // Tail rows (valid == false) still join every full-mask shuffle.
        float total = half_sum(x[0]);
        CUTLASS_PRAGMA_UNROLL
        for (int h = 1; h < kHalves; ++h) total = add(total, half_sum(x[h]));
        const float inv = inverse(total, float(D), p.eps);
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < kHalves; ++h) {
          const int col = start_column + h * 128;
          Vector y;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 8; ++i) y[i] = BF(mul(mul(float(x[h][i]), inv), p.weight[col + i]));
          if (valid) cutlass::arch::global_store<Vector, 16>(y, p.norm + row * D + col, true);
        }
      });
    }
  };

  CUTLASS_DEVICE void operator()(Op const& op, Iterator destination, typename Base::AccumulatorTile const& acc,
                                 Iterator source) {
    Source aspect{{source, destination, {}, {}}};
    Base::template operator()<Source>(op, destination, acc, aspect);
  }
};

template <int D>
using Gemm = cutlass::gemm::device::Gemm<
    BF, cutlass::layout::RowMajor, BF, cutlass::layout::ColumnMajor, BF, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, D, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, Op,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 3, 8, 8, false, cutlass::arch::OpMultiplyAdd>;

template <int D>
using Kernel = cutlass::gemm::kernel::Gemm<typename Gemm<D>::GemmKernel::Mma,
                                           ColumnProjectionRms<D, typename Gemm<D>::GemmKernel::Epilogue>,
                                           typename Gemm<D>::GemmKernel::ThreadblockSwizzle, false>;

constexpr int64_t kMaxM = INT_MAX - 128;  // the swizzle computes M + tile_m in int32

template <int D>
void launch(at::Tensor const& a, at::Tensor const& w, at::Tensor const& bias, at::Tensor const& dest, Op::Params p) {
  using G = Gemm<D>;
  using K = Kernel<D>;
  static_assert(sizeof(typename K::SharedStorage) == sizeof(typename G::GemmKernel::SharedStorage), "no extra smem");
  auto bf = [](at::Tensor const& t) { return reinterpret_cast<BF*>(t.data_ptr()); };
  typename G::Arguments args({int(a.size(0)), D, D}, {bf(a), D}, {bf(w), D}, {bf(bias), 0}, {bf(dest), D}, p, 1);
  TORCH_CHECK(G().can_implement(args) == cutlass::Status::kSuccess,
              "projection_rms: CUTLASS cannot implement this GEMM");
  typename G::GemmKernel::ThreadblockSwizzle swizzle;
  const auto tiled = swizzle.get_tiled_shape(args.problem_size, {128, D, 32}, 1);
  typename K::Params params(args.problem_size, tiled, args.ref_A.non_const_ref(), args.ref_B.non_const_ref(),
                            args.ref_C.non_const_ref(), args.ref_D, args.epilogue, nullptr);
  constexpr int smem = sizeof(typename K::SharedStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cutlass::Kernel<K>
      <<<swizzle.get_grid_shape(tiled), K::kThreadCount, smem, at::cuda::getCurrentCUDAStream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The Python supports() decides eligibility; these checks only keep a bad call from touching memory.
void run(at::Tensor const& a, at::Tensor const& w, at::Tensor const& bias, at::Tensor const& table,
         at::Tensor const& weight, at::Tensor const& dest, at::Tensor const& norm, double eps) {
  const int64_t m = a.size(0), d = a.size(1);
  TORCH_CHECK(a.dim() == 2 && (d == 128 || d == 256) && m >= 1 && m <= kMaxM,
              "projection_rms: a must be [M, 128] or [M, 256] with 1 <= M <= INT_MAX - 128");
  TORCH_CHECK(table.dim() == 3 && table.size(2) == d && w.sizes() == at::IntArrayRef({d, d}) && bias.size(0) == d &&
                  weight.size(0) == d && dest.numel() == m * d && norm.numel() == m * d,
              "projection_rms: w, bias, weight and table must match the width of a; dest and norm its size");
  for (auto const* t : {&a, &w, &bias, &table, &dest, &norm})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() &&
                    t->data_ptr<at::BFloat16>() && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "projection_rms: a, w, bias, table, dest and norm must be contiguous 16-byte aligned CUDA bf16");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kFloat && weight.is_contiguous(),
              "projection_rms: weight must be a contiguous CUDA fp32 tensor");
  Op::Params p;
  p.table = reinterpret_cast<BF const*>(table.data_ptr());
  p.weight = weight.data_ptr<float>();
  p.norm = reinterpret_cast<BF*>(norm.data_ptr());
  p.rows = m;
  p.contexts = table.size(0);
  p.row_features = table.size(1);
  p.prefix = table.size(1) - m / table.size(0);
  p.eps = float(eps);
  TORCH_CHECK(m % table.size(0) == 0 && p.prefix >= 0,
              "projection_rms: a must have C * F rows with F <= table.size(1)");
  const c10::cuda::CUDAGuard guard(a.device());
  if (d == 128)
    launch<128>(a, w, bias, dest, p);
  else
    launch<256>(a, w, bias, dest, p);
}
}  // namespace projection_rms

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &projection_rms::run); }
