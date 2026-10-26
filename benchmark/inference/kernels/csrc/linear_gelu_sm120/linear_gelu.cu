// out = gelu(bf16(a @ w.T + bias)) for BF16 a[M,K], w[N,K], bias[N], out[M,N]; SM120.
// Matches cuBLAS addmm (FP32 accumulate, bias added in FP32, RN to BF16) followed by Inductor's GELU.
#include <climits>
#include <type_traits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/gemm/device/gemm.h>
#include "common/bias_gelu.cuh"

namespace linear_gelu {
using BF = cutlass::bfloat16_t;

struct BiasGelu : bias_gelu::Op {  // names this module in the kernel symbol
  using bias_gelu::Op::Op;
};

template <int TileN, class Op = BiasGelu>
using Gemm = cutlass::gemm::device::Gemm<
    BF, cutlass::layout::RowMajor, BF, cutlass::layout::ColumnMajor, BF, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, TileN, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, Op,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 3, 8, 8, false, cutlass::arch::OpMultiplyAdd>;

constexpr int64_t kMaxM = INT_MAX - 128;  // the swizzle computes M + tile_m in int32

BF* ptr(at::Tensor const& t) { return reinterpret_cast<BF*>(t.data_ptr<at::BFloat16>()); }

template <int K, int N, int TileN>
void launch(at::Tensor const& a, at::Tensor const& w, at::Tensor const& bias, at::Tensor const& out) {
  static_assert(K % 8 == 0 && N % 8 == 0, "16-byte vector access");
  using G = Gemm<TileN>;
  typename G::Arguments args({int(a.size(0)), N, K}, {ptr(a), K}, {ptr(w), K}, {ptr(bias), 0}, {ptr(out), N},
                             {1.0f, 1.0f}, 1);
  G gemm;
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "linear_gelu: can_implement failed");
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(gemm.initialize(args, nullptr, stream) == cutlass::Status::kSuccess, "linear_gelu: initialize failed");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "linear_gelu: run failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// K1024/N2048 (dataset FFN): the same GEMM with an epilogue that loads the bias through the source aspect and
// applies bias and GELU to each 8-element vector of the accumulator exchange.
using Bias = cutlass::epilogue::thread::LinearCombination<BF, 8, float, float>;
struct FfnOp : Bias {
  struct Params : Bias::Params {
    CUTLASS_HOST_DEVICE Params() : Bias::Params(1.f, 1.f) {}
  };
  CUTLASS_HOST_DEVICE FfnOp(Params const& p) : Bias(p) {}
};

template <class Base>
struct GeluEpilogue : Base {
  using Iterator = typename Base::OutputTileIterator;
  using Fragment = typename Iterator::Fragment;
  using SharedFragment = typename Base::SharedLoadIterator::Fragment;
  static_assert(Iterator::kElementsPerAccess == 8, "8-element vectors");
  static_assert(Fragment::kElements == SharedFragment::kElements, "one output fragment per exchanged fragment");
  CUTLASS_DEVICE GeluEpilogue(typename Base::SharedStorage& s, int t, int w, int l) : Base(s, t, w, l) {}

  struct Source {
    Iterator bias_iterator;
    Fragment bias;
    CUTLASS_DEVICE void load() {
      bias_iterator.load(bias);
      ++bias_iterator;
    }
    CUTLASS_DEVICE void apply_output_operator(Fragment& out, FfnOp const& op, SharedFragment const& acc) {
      auto const* av = reinterpret_cast<typename Bias::FragmentAccumulator const*>(&acc);
      auto const* bv = reinterpret_cast<typename Bias::FragmentSource const*>(&bias);
      auto* ov = reinterpret_cast<typename Bias::FragmentOutput*>(&out);
      CUTLASS_PRAGMA_UNROLL
      for (int idx = 0; idx < Fragment::kElements / 8; ++idx) {
        auto rounded = static_cast<Bias const&>(op)(av[idx], bv[idx]);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) {
          BF scalar = rounded[i];
          auto value = __ushort_as_bfloat16(scalar.raw());
          ov[idx][i] = BF::bitcast(__bfloat16_as_ushort(triton_gelu_bf16(value)));
        }
      }
    }
  };

  CUTLASS_DEVICE void operator()(FfnOp const& op, Iterator destination, typename Base::AccumulatorTile const& acc,
                                 Iterator source) {
    Source aspect{source, {}};
    Base::template operator()<Source>(op, destination, acc, aspect);
  }
};

using FfnGemm = Gemm<256, FfnOp>;
using FfnBase = FfnGemm::GemmKernel;
using FfnKernel = cutlass::gemm::kernel::Gemm<typename FfnBase::Mma, GeluEpilogue<typename FfnBase::Epilogue>,
                                              typename FfnBase::ThreadblockSwizzle, false>;
static_assert(sizeof(FfnKernel::SharedStorage) == sizeof(FfnBase::SharedStorage), "no extra shared memory");

void launch_ffn(at::Tensor const& a, at::Tensor const& w, at::Tensor const& bias, at::Tensor const& out) {
  FfnGemm::Arguments args({int(a.size(0)), 2048, 1024}, {ptr(a), 1024}, {ptr(w), 1024}, {ptr(bias), 0},
                          {ptr(out), 2048}, FfnOp::Params(), 1);
  TORCH_CHECK(FfnGemm::can_implement(args) == cutlass::Status::kSuccess, "linear_gelu: can_implement failed");
  typename FfnBase::ThreadblockSwizzle swizzle;
  auto tiled = swizzle.get_tiled_shape(args.problem_size, {128, 256, 32}, 1);
  FfnKernel::Params params(args.problem_size, tiled, args.ref_A.non_const_ref(), args.ref_B.non_const_ref(),
                           args.ref_C.non_const_ref(), args.ref_D, args.epilogue, nullptr);
  constexpr int smem = sizeof(FfnKernel::SharedStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<FfnKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cutlass::Kernel<FfnKernel>
      <<<swizzle.get_grid_shape(tiled), FfnKernel::kThreadCount, smem, at::cuda::getCurrentCUDAStream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

struct Instance {
  int64_t k, n;
  void (*fn)(at::Tensor const&, at::Tensor const&, at::Tensor const&, at::Tensor const&);
};
constexpr Instance kInstances[] = {
    {128, 256, launch<128, 256, 128>},  // embedding MLPs, Small
    {256, 512, launch<256, 512, 256>},  // embedding MLPs, Large and Medium
    {1024, 2048, launch_ffn},           // dataset FFN, Large
};

// The Python supports() is the contract; these checks only keep a bad call from touching memory.
void run(at::Tensor const& a, at::Tensor const& w, at::Tensor const& bias, at::Tensor const& out) {
  for (auto const* t : {&a, &w, &bias, &out})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "linear_gelu: operands must be contiguous, 16-byte aligned CUDA BF16 tensors");
  const int64_t m = a.size(0), k = a.size(1), n = w.size(0);
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && w.size(1) == k && bias.dim() == 1 && bias.size(0) == n &&
                  out.dim() == 2 && out.size(0) == m && out.size(1) == n && m >= 1 && m <= kMaxM,
              "linear_gelu: expected a [M, K], w [N, K], bias [N], out [M, N]");
  c10::cuda::CUDAGuard guard(a.device());
  for (auto const& i : kInstances)
    if (i.k == k && i.n == n) return i.fn(a, w, bias, out);
  TORCH_CHECK(false, "linear_gelu: no instance for K=", k, " N=", n);
}
}  // namespace linear_gelu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &linear_gelu::run); }
