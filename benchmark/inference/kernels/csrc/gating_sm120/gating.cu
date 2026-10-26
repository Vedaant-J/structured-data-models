// Row-block GatedLogScale gate MLP for head width H (H = 64 Large/Medium, 32 Small), SM120, BF16.
//   first:  h   = gelu(bf16(q @ w1.T + b1))                          q[M,H]  w1[64,H]  h[M,64]
//   second: out = bf16(q * logscale * scale[row%4] * (1 + tanh(bf16(h @ w2.T + b2))))   w2[H,64]  out[M,H]
// Both GEMMs are the CUTLASS form of the kernel cuBLAS picks for them at model sizes, including K32
// (cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_align8): one 32x32x64 CTA tile, K16 BF16
// mma_sync groups accumulated in ascending K in FP32, bias added in FP32, rounded to BF16 before the
// activation. At K32 the K64 tile reads K 32..63 as predicated zeros, as the cuBLAS kernel does. (At tiny M
// cuBLAS picks its 32x32_128x2 variant for the second GEMM; that gives the same bytes, see the tests.)
#include <climits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "common/bf16_wmma.cuh"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include "common/bias_gelu.cuh"

namespace gating {
using BF = cutlass::bfloat16_t;
using Linear = cutlass::epilogue::thread::LinearCombination<BF, 8, float, float>;
constexpr int kHidden = 64, kHeads = 4;
constexpr int64_t kMaxM = INT_MAX - 32;  // the swizzle computes M + 31 in int32

template <class Op>
using Gemm = cutlass::gemm::device::Gemm<
    BF, cutlass::layout::RowMajor, BF, cutlass::layout::ColumnMajor, BF, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassWmmaTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 16, 64>, cutlass::gemm::GemmShape<16, 16, 16>, Op,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 1, 8, 8, false, cutlass::arch::OpMultiplyAdd>;

struct BiasGelu : bias_gelu::Op {  // names this module in the kernel symbol
  using bias_gelu::Op::Op;
};

// Inductor's libdevice tanh as emitted in the native gate kernel's PTX (explicit FTZ where it has FTZ).
__device__ __forceinline__ float native_tanh(float x) {
  float result;
  asm("{ .reg .pred p; .reg .f32 a,y,t,e,d,r,v; .reg .b32 s,b;\n"
      "abs.ftz.f32 a,%1;\n"
      "setp.ltu.f32 p,a,0f3F19999A;\n"
      "@!p bra large;\n"
      "mul.f32 y,%1,%1;\n"
      "fma.rn.ftz.f32 t,0f3C80F082,y,0fBD563CAE;\n"
      "fma.rn.ftz.f32 t,t,y,0f3E085941;\n"
      "fma.rn.ftz.f32 t,t,y,0fBEAAA9ED;\n"
      "fma.rn.ftz.f32 t,t,y,0f00000000;\n"
      "fma.rn.ftz.f32 %0,t,%1,%1;\n"
      "bra done;\n"
      "large:\n"
      "mul.f32 e,a,0f4038AA3B;\n"
      "ex2.approx.ftz.f32 e,e;\n"
      "add.f32 d,e,0f3F800000;\n"
      "rcp.approx.ftz.f32 r,d;\n"
      "fma.rn.ftz.f32 v,r,0fC0000000,0f3F800000;\n"
      "setp.ge.f32 p,a,0f41102CB4;\n"
      "selp.f32 v,0f3F800000,v,p;\n"
      "mov.b32 s,%1; and.b32 s,s,0x80000000;\n"
      "mov.b32 b,v; or.b32 b,b,s; mov.b32 %0,b;\n"
      "done: }\n"
      : "=f"(result)
      : "f"(x));
  return result;
}

// ((q * logscale) * scale) * (tanh(z) + 1), each an RN FP32 op without contraction, then RN BF16.
__device__ __forceinline__ BF gate(BF z, BF q, float scale, float logscale) {
  float a, b, c, r;
  const float t = native_tanh(float(z));
  asm("mul.rn.f32 %0,%1,%2;" : "=f"(a) : "f"(float(q)), "f"(logscale));
  asm("mul.rn.f32 %0,%1,%2;" : "=f"(b) : "f"(a), "f"(scale));
  asm("add.rn.f32 %0,%1,0f3F800000;" : "=f"(c) : "f"(t));
  asm("mul.rn.f32 %0,%1,%2;" : "=f"(r) : "f"(b), "f"(c));
  return BF(r);
}

struct GateOp : Linear {
  struct Params : Linear::Params {
    BF const* q;
    float const* scale;
    int rows;
    float logscale;
    CUTLASS_HOST_DEVICE Params(BF const* q = nullptr, float const* scale = nullptr, int rows = 0, float logscale = 0)
        : Linear::Params(1.0f, 1.0f), q(q), scale(scale), rows(rows), logscale(logscale) {}
  };
  Params p;
  CUTLASS_HOST_DEVICE GateOp(Params const& p) : Linear(p), p(p) {}
};

// The stock epilogue (accumulator exchange, bias load, predicated store) with a source aspect that also
// loads Q at the destination coordinates, so the operator sees z, Q and the output row together.
template <class Base, int N>
struct GateEpilogue : Base {
  using Iterator = typename Base::OutputTileIterator;
  using Fragment = typename Iterator::Fragment;
  using SharedFragment = typename Base::SharedLoadIterator::Fragment;
  using ThreadMap = typename Iterator::ThreadMap;
  static_assert(Iterator::kIterations == 1 && Fragment::kElements == 8 && SharedFragment::kElements == 8 &&
                    ThreadMap::Iterations::kCount == 1,
                "one eight-element output vector per thread");
  int thread;
  CUTLASS_DEVICE GateEpilogue(typename Base::SharedStorage& s, int t, int w, int lane)
      : Base(s, t, w, lane), thread(t) {}

  struct Source {
    Iterator bias_it, q_it;
    Fragment bias, q;
    int row;
    CUTLASS_DEVICE void load() {
      bias_it.load(bias);
      q_it.load(q);
      ++bias_it;
      ++q_it;
    }
    CUTLASS_DEVICE void apply_output_operator(Fragment& out, GateOp const& op, SharedFragment const& acc) {
      auto const& values = *reinterpret_cast<typename Linear::FragmentAccumulator const*>(&acc);
      auto const& source = *reinterpret_cast<typename Linear::FragmentOutput const*>(&bias);
      const typename Linear::FragmentOutput z = static_cast<Linear const&>(op)(values, source);
      const float scale = op.p.scale[row % kHeads];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 8; ++i) out[i] = gate(BF(z[i]), BF(q[i]), scale, op.p.logscale);
    }
  };

  CUTLASS_DEVICE void operator()(GateOp const& op, Iterator destination, typename Base::AccumulatorTile const& acc,
                                 Iterator bias) {
    typename Iterator::Params q_params{cutlass::layout::RowMajor(N)};
    auto offset = destination.thread_start() - ThreadMap::initial_offset(thread);
    Iterator q_it(q_params, const_cast<BF*>(op.p.q), {op.p.rows, N}, thread, offset);
    Source aspect{bias, q_it, {}, {}, destination.thread_start().row()};
    Base::template operator()<Source>(op, destination, acc, aspect);
  }
};

template <int H>
struct Head {
  static_assert(H % 16 == 0 && H <= 64, "head width");
  using First = Gemm<BiasGelu>;
  using SecondBase = typename Gemm<GateOp>::GemmKernel;
  using Second = cutlass::gemm::kernel::Gemm<typename SecondBase::Mma, GateEpilogue<typename SecondBase::Epilogue, H>,
                                             typename SecondBase::ThreadblockSwizzle, false>;
  static_assert(sizeof(typename Second::SharedStorage) < 48 * 1024, "static shared launch");

  static void first(at::Tensor const& q, at::Tensor const& w, at::Tensor const& b, at::Tensor const& out) {
    typename First::Arguments args({int(q.size(0)), kHidden, H}, {ptr(q), H}, {ptr(w), H}, {ptr(b), 0},
                                   {ptr(out), kHidden}, {1.0f, 1.0f}, 1);
    First gemm;
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "gating: first layer can_implement failed");
    TORCH_CHECK(gemm.initialize(args, nullptr, stream) == cutlass::Status::kSuccess,
                "gating: first layer initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "gating: first layer run failed");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  static void second(at::Tensor const& h, at::Tensor const& w, at::Tensor const& b, at::Tensor const& q,
                     at::Tensor const& scale, at::Tensor const& out, double logscale) {
    const int m = int(h.size(0));
    const cutlass::gemm::GemmCoord size(m, H, kHidden);
    typename SecondBase::ThreadblockSwizzle swizzle;
    auto tiled = swizzle.get_tiled_shape(size, {32, 32, kHidden}, 1);
    typename Second::Params params(size, tiled, {ptr(h), kHidden}, {ptr(w), kHidden}, {ptr(b), 0}, {ptr(out), H},
                                   GateOp::Params(ptr(q), scale.data_ptr<float>(), m, float(logscale)), nullptr);
    cutlass::Kernel<Second><<<swizzle.get_grid_shape(tiled), Second::kThreadCount,
                              sizeof(typename Second::SharedStorage), at::cuda::getCurrentCUDAStream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  static BF* ptr(at::Tensor const& t) { return reinterpret_cast<BF*>(t.data_ptr<at::BFloat16>()); }
};

void check(std::initializer_list<at::Tensor const*> ts) {
  for (auto const* t : ts)
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "gating: operands must be contiguous, 16-byte aligned CUDA BF16 tensors");
}

// The Python supports() is the contract; these checks only keep a bad call from touching memory.
void first(at::Tensor const& q, at::Tensor const& w, at::Tensor const& b, at::Tensor const& out) {
  check({&q, &w, &b, &out});
  const int64_t m = q.size(0), k = q.size(1);
  TORCH_CHECK(q.dim() == 2 && m >= 1 && m <= kMaxM && w.dim() == 2 && w.size(0) == kHidden && w.size(1) == k &&
                  b.dim() == 1 && b.size(0) == kHidden && out.dim() == 2 && out.size(0) == m && out.size(1) == kHidden,
              "gating: expected q [M, H], w1 [64, H], b1 [64], h [M, 64]");
  c10::cuda::CUDAGuard guard(q.device());
  if (k == 64) return Head<64>::first(q, w, b, out);
  if (k == 32) return Head<32>::first(q, w, b, out);
  TORCH_CHECK(false, "gating: no first-layer instance for head ", k);
}

void second(at::Tensor const& h, at::Tensor const& w, at::Tensor const& b, at::Tensor const& q, at::Tensor const& scale,
            at::Tensor const& out, double logscale) {
  check({&h, &w, &b, &q, &out});
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == at::kFloat && scale.is_contiguous() && scale.numel() == kHeads,
              "gating: scale must be a contiguous CUDA float32 tensor of 4 heads");
  const int64_t m = h.size(0), n = q.size(1);
  TORCH_CHECK(h.dim() == 2 && m >= 1 && m <= kMaxM && h.size(1) == kHidden && w.dim() == 2 && w.size(0) == n &&
                  w.size(1) == kHidden && b.dim() == 1 && b.size(0) == n && q.dim() == 2 && q.size(0) == m &&
                  out.dim() == 2 && out.size(0) == m && out.size(1) == n,
              "gating: expected h [M, 64], w2 [H, 64], b2 [H], q [M, H], out [M, H]");
  c10::cuda::CUDAGuard guard(h.device());
  if (n == 64) return Head<64>::second(h, w, b, q, scale, out, logscale);
  if (n == 32) return Head<32>::second(h, w, b, q, scale, out, logscale);
  TORCH_CHECK(false, "gating: no second-layer instance for head ", n);
}
}  // namespace gating

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("first", &gating::first);
  m.def("second", &gating::second);
}
