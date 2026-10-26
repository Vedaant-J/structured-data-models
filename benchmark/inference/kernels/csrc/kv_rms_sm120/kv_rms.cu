// Column KV projection with the key RMS fused into the K epilogue; raw K is never stored. SM120, BF16.
//
//   packed[:, D:] = bf16(a @ w[D:].T + bias[D:])                          (V)
//   key           = rms(bf16(a @ w[:D].T + bias[:D])) per head of Head    (normalized K, no weight, eps 1e-6)
//
// a [M, D], w [2D, D], bias [2D], packed [M, 2D] (packed[:, :D] is left untouched), key [M, D].
// key may be exactly a (the compiled graph reuses A's buffer for normalized K); nothing else may overlap.
// Two launches on one stream, each a single N tile of width D: V first, then K. A K CTA overwrites only
// the rows of A it alone has read, so the alias is safe without inter-CTA synchronization.
// Bytes match cuBLAS addmm followed by SDM's Triton rmsnorm_cast at channels = Head (32 or 64).
#include <climits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "common/head_rms.cuh"

namespace kv_rms {
using head_rms::BF;
using head_rms::kMaxM;
using head_rms::launch;
struct Key {};  // names the K launch in the kernel symbol

template <int D, int Head>
void run_instance(at::Tensor const& a, at::Tensor const& w, at::Tensor const& bias, at::Tensor const& packed,
                  at::Tensor const& key) {
  auto ptr = [](at::Tensor const& t) { return reinterpret_cast<BF*>(t.data_ptr<at::BFloat16>()); };
  const int m = int(a.size(0));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch<D, Head, false, Key>(ptr(a), ptr(w) + D * D, ptr(bias) + D, ptr(packed) + D, 2 * D, m, stream);
  launch<D, Head, true, Key>(ptr(a), ptr(w), ptr(bias), ptr(key), D, m, stream);
}

// The Python supports() decides eligibility (including aliasing); these checks only keep a bad call from touching
// memory.
void run(at::Tensor const& a, at::Tensor const& w, at::Tensor const& bias, at::Tensor const& packed,
         at::Tensor const& key, int64_t head) {
  for (auto const* t : {&a, &w, &bias, &packed, &key})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "kv_rms: a, w, bias, packed and key must be contiguous 16-byte aligned CUDA bf16");
  const int64_t m = a.size(0), d = a.size(1);
  TORCH_CHECK(a.dim() == 2 && m >= 1 && m <= kMaxM && w.dim() == 2 && w.size(0) == 2 * d && w.size(1) == d &&
                  bias.dim() == 1 && bias.size(0) == 2 * d && packed.dim() == 2 && packed.size(0) == m &&
                  packed.size(1) == 2 * d && key.sizes() == a.sizes(),
              "kv_rms: need a [M, D], w [2D, D], bias [2D], packed [M, 2D] and key [M, D]");
  c10::cuda::CUDAGuard guard(a.device());
  if (d == 256 && head == 64) return run_instance<256, 64>(a, w, bias, packed, key);
  if (d == 128 && head == 32) return run_instance<128, 32>(a, w, bias, packed, key);
  TORCH_CHECK(false, "kv_rms: (D, head) must be (256, 64) or (128, 32), got (", d, ", ", head, ")");
}
}  // namespace kv_rms

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &kv_rms::run); }
