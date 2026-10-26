// Column query projection with SDM's unweighted per-head RMS fused into the epilogue; the raw query is
// never stored. SM120, BF16.
//
//   out = rms(bf16(a @ w.T + bias)) per head of Head channels, eps 1e-6, no weight
//
// a [M, D], w [D, D], bias [D], out [M, D]; out may be exactly a, nothing else may overlap it.
// One launch of common/head_rms.cuh with a single N tile of width D.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "common/head_rms.cuh"

namespace query_rms {
using head_rms::BF;
struct Query {};  // names this launch in the kernel symbol

template <int D, int Head>
void run_instance(at::Tensor const& a, at::Tensor const& w, at::Tensor const& bias, at::Tensor const& out) {
  auto ptr = [](at::Tensor const& t) { return reinterpret_cast<BF*>(t.data_ptr<at::BFloat16>()); };
  head_rms::launch<D, Head, true, Query>(ptr(a), ptr(w), ptr(bias), ptr(out), D, int(a.size(0)),
                                         at::cuda::getCurrentCUDAStream());
}

// The Python supports() decides eligibility (including aliasing); these checks only keep a bad call from touching
// memory.
void run(at::Tensor const& a, at::Tensor const& w, at::Tensor const& bias, at::Tensor const& out, int64_t head) {
  for (auto const* t : {&a, &w, &bias, &out})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "query_rms: a, w, bias and out must be contiguous 16-byte aligned CUDA bf16");
  const int64_t m = a.size(0), d = a.size(1);
  TORCH_CHECK(a.dim() == 2 && m >= 1 && m <= head_rms::kMaxM && w.dim() == 2 && w.size(0) == d && w.size(1) == d &&
                  bias.dim() == 1 && bias.size(0) == d && out.sizes() == a.sizes(),
              "query_rms: need a [M, D], w [D, D], bias [D] and out [M, D]");
  c10::cuda::CUDAGuard guard(a.device());
  if (d == 256 && head == 64) return run_instance<256, 64>(a, w, bias, out);
  if (d == 128 && head == 32) return run_instance<128, 32>(a, w, bias, out);
  TORCH_CHECK(false, "query_rms: (D, head) must be (256, 64) or (128, 32), got (", d, ", ", head, ")");
}
}  // namespace query_rms

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &query_rms::run); }
