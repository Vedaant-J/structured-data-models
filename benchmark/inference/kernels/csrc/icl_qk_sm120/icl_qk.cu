// Per-head RMS of the ICL attention query and key, with the cache copies fused in. SM120, BF16, head width 64.
//
//   query: out[m, h, :] = bf16(bf16(rms(q[m, h, :])) * log_n * head_scale[h])   (FP32 products, one rounding)
//   key:   k[m, h, :]       = rms(kv[m, h * 64 : (h + 1) * 64])                   for h < key_heads
//          k_cache[m, h, :] = rms(kv[m, h * 64 : ...])                            for h < cache
//          v_cache[m, h, :] = kv[m, (heads + h) * 64 : ...]                       for h < cache
//
// q [M, H, 64] with strides (ld, 64, 1); kv [M, 2 * heads * 64] with row stride ld (packed K | V of the KV addmm,
// read in place). head_scale FP32 [H]. Outputs contiguous. eps 1e-6, no weight.
// Bytes match SDM's Triton rmsnorm_cast at channels = 64, then Inductor's fused LogScale multiply (FP32 between
// the two products, head_scale read as FP32) and the cache head-slice copies.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "common/rms_ptx.cuh"

namespace icl_qk {
using rms_ptx::add;
using rms_ptx::inv_rms;
using rms_ptx::mul;
using rms_ptx::square_add;
constexpr int kHead = 64, kLanes = 8, kThreads = 256, kUnits = kThreads / kLanes;
using u16 = unsigned short;

__device__ __forceinline__ float f32(u16 a) {
  float x;
  asm("cvt.f32.bf16 %0,%1;" : "=f"(x) : "h"(a));
  return x;
}
__device__ __forceinline__ u16 bf16(float a) {
  u16 x;
  asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(x) : "f"(a));
  return x;
}
__device__ __forceinline__ u16 element(uint4 w, int i) {
  const unsigned word = i < 2 ? w.x : i < 4 ? w.y : i < 6 ? w.z : w.w;
  return u16(word >> (16 * (i & 1)));
}
__device__ __forceinline__ uint4 pack(const u16 (&a)[8]) {
  return make_uint4(a[0] | unsigned(a[1]) << 16, a[2] | unsigned(a[3]) << 16, a[4] | unsigned(a[5]) << 16,
                    a[6] | unsigned(a[7]) << 16);
}

// Each lane holds 8 adjacent channels (two rmsnorm_cast groups of 4); 8 adjacent lanes hold one head.
// rmsnorm_cast squares each group in channel order and reduces the 16 group sums by an XOR butterfly with group
// shifts 8, 4, 2, 1: lane shifts 4, 2, 1 are group shifts 8, 4, 2 and the in-lane add of the two groups is group
// shift 1. All lanes, including those of masked units, take part in every shuffle.
__device__ __forceinline__ void head_rms(uint4 w, u16 (&out)[8]) {
  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    s0 = square_add(element(w, i), s0);
    s1 = square_add(element(w, i + 4), s1);
  }
#pragma unroll
  for (int shift = kLanes / 2; shift >= 1; shift >>= 1) {
    s0 = add(s0, __shfl_xor_sync(~0u, s0, shift, kLanes));
    s1 = add(s1, __shfl_xor_sync(~0u, s1, shift, kLanes));
  }
  const float scale = inv_rms<kHead>(add(s0, s1));
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = bf16(mul(f32(element(w, i)), scale));
}

__global__ void __launch_bounds__(kThreads)
    query_kernel(const u16* __restrict__ q, int64_t ld, int heads, const float* __restrict__ head_scale, float log_n,
                 u16* __restrict__ out, int64_t units) {
  const int lane = threadIdx.x % kLanes;
  const int64_t unit = int64_t(blockIdx.x) * kUnits + threadIdx.x / kLanes;
  const bool valid = unit < units;
  const int64_t row = unit / heads;
  const int h = int(unit % heads);
  const uint4 w = valid ? *reinterpret_cast<const uint4*>(q + row * ld + h * kHead + lane * 8) : make_uint4(0, 0, 0, 0);
  u16 r[8];
  head_rms(w, r);
  if (!valid) return;
  const float a = head_scale[h];
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = bf16(mul(mul(f32(r[i]), log_n), a));
  *reinterpret_cast<uint4*>(out + unit * kHead + lane * 8) = pack(r);
}

// Units cover heads [0, span) of each row: span = heads when K is stored, else cache.
__global__ void __launch_bounds__(kThreads)
    key_kernel(const u16* __restrict__ kv, int64_t ld, int heads, int span, bool key, int cache,
               u16* __restrict__ k_out, u16* __restrict__ k_cache, u16* __restrict__ v_cache, int64_t units) {
  const int lane = threadIdx.x % kLanes;
  const int64_t unit = int64_t(blockIdx.x) * kUnits + threadIdx.x / kLanes;
  const bool valid = unit < units;
  const int64_t row = unit / span;
  const int h = int(unit % span);
  const u16* src = kv + row * ld + h * kHead + lane * 8;
  const uint4 w = valid ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
  u16 r[8];
  head_rms(w, r);
  if (!valid) return;
  const uint4 packed = pack(r);
  if (key) *reinterpret_cast<uint4*>(k_out + (row * heads + h) * kHead + lane * 8) = packed;
  if (h < cache) {
    const int64_t at = (row * cache + h) * kHead + lane * 8;
    *reinterpret_cast<uint4*>(k_cache + at) = packed;
    *reinterpret_cast<uint4*>(v_cache + at) = *reinterpret_cast<const uint4*>(src + heads * kHead);
  }
}

u16* ptr(at::Tensor const& t) { return reinterpret_cast<u16*>(t.data_ptr()); }
int64_t blocks(int64_t units) { return (units + kUnits - 1) / kUnits; }

// The Python supports() decides eligibility; these checks only keep a bad call from touching memory.
void check_rows(at::Tensor const& t, int64_t width, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.stride(-1) == 1 && t.size(0) > 0 &&
                  t.stride(0) % 8 == 0 && t.stride(0) >= width && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              "icl_qk: ", what, " must be CUDA bf16 rows of at least ", width, " elements, 16-byte aligned");
}

at::Tensor query(at::Tensor const& q, at::Tensor const& head_scale, double log_n) {
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kHead && q.stride(1) == kHead,
              "icl_qk: q must be [M, H, 64] with head stride 64");
  check_rows(q, q.size(1) * kHead, "q");
  TORCH_CHECK(head_scale.is_cuda() && head_scale.scalar_type() == at::kFloat && head_scale.is_contiguous() &&
                  head_scale.numel() == q.size(1),
              "icl_qk: head_scale must be a contiguous CUDA fp32 [H] tensor");
  c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty(q.sizes(), q.options().memory_format(at::MemoryFormat::Contiguous));
  const int64_t units = q.size(0) * q.size(1);
  TORCH_CHECK(blocks(units) <= INT_MAX, "icl_qk: q has too many rows for an int32 grid");
  query_kernel<<<blocks(units), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      ptr(q), q.stride(0), int(q.size(1)), head_scale.data_ptr<float>(), float(log_n), ptr(out), units);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> key(at::Tensor const& kv, int64_t heads, bool with_key, int64_t cache) {
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 2 * heads * kHead, "icl_qk: kv must be [M, 2 * heads * 64]");
  TORCH_CHECK(0 <= cache && cache <= heads && (with_key || cache > 0),
              "icl_qk: cache must be in [0, heads] and nonzero without with_key");
  check_rows(kv, kv.size(1), "kv");
  c10::cuda::CUDAGuard guard(kv.device());
  const int64_t m = kv.size(0);
  auto opts = kv.options().memory_format(at::MemoryFormat::Contiguous);
  auto k_out = at::empty({m, with_key ? heads : 0, kHead}, opts);
  auto k_cache = at::empty({m, cache, kHead}, opts), v_cache = at::empty({m, cache, kHead}, opts);
  const int span = int(with_key ? heads : cache);
  const int64_t units = m * span;
  TORCH_CHECK(blocks(units) <= INT_MAX, "icl_qk: kv has too many rows for an int32 grid");
  key_kernel<<<blocks(units), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      ptr(kv), kv.stride(0), int(heads), span, with_key, int(cache), ptr(k_out), ptr(k_cache), ptr(v_cache), units);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {k_out, k_cache, v_cache};
}
}  // namespace icl_qk

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("query", &icl_qk::query);
  m.def("key", &icl_qk::key);
}
