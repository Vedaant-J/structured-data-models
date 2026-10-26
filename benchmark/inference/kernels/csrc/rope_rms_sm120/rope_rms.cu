// Split-half RoPE followed by the unweighted per-head RMS of Q and of K, one launch. SM120, BF16.
//
//   q_out[c, s, h, :] = rms(rope(q[c, s, h, :], s)),  k_out likewise from k   (eps 1e-6, no weight)
//   rope(x, s)[j]     = bf16(fma(x[j], cos(a), -x[j + Head/2] * sin(a)))   for j <  Head/2
//                       bf16(fma(x[j], cos(a),  x[j - Head/2] * sin(a)))   for j >= Head/2
//   a = s * inv_freq[j % (Head/2)]
//
// q, k: [C, S, 4, Head] BF16 with strides (S * ld, ld, Head, 1); ld may differ between q and k (K is a slice of
// packed KV). inv_freq: [Head / 2] FP32. q_out, k_out: contiguous [C, S, 4, Head], disjoint from the inputs.
// Bytes match Inductor's fused rope pointwise (FP32 between ops, libdevice sin/cos, one BF16 rounding) followed
// by SDM's Triton rmsnorm_cast at channels = Head.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "common/rms_ptx.cuh"

namespace rope_rms {
using rms_ptx::add;
using rms_ptx::inv_rms;
using rms_ptx::mul;
using rms_ptx::square_add;
constexpr int kHeads = 4, kThreads = 128;

__device__ __forceinline__ float bf16_to_f32(unsigned short a) {
  float x;
  asm("cvt.f32.bf16 %0,%1;" : "=f"(x) : "h"(a));
  return x;
}
__device__ __forceinline__ unsigned short f32_to_bf16(float a) {
  unsigned short x;
  asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(x) : "f"(a));
  return x;
}
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float x;
  asm("fma.rn.f32 %0,%1,%2,%3;" : "=f"(x) : "f"(a), "f"(b), "f"(c));
  return x;
}
// Triton's libdevice sin/cos run with FTZ, so a subnormal angle is first flushed to a signed zero.
// Checked bitwise against Inductor's tl_math.sin/cos over every FP32 input.
__device__ __forceinline__ void trig(float a, float& s, float& c) {
  asm("mul.ftz.f32 %0,%0,0f3F800000;" : "+f"(a));
  s = sinf(a);
  c = cosf(a);
}
// Element i of 4 BF16 values packed in w.
__device__ __forceinline__ float element(uint2 w, int i) { return bf16_to_f32((i < 2 ? w.x : w.y) >> (16 * (i & 1))); }
__device__ __forceinline__ uint2 pack(const unsigned short (&a)[4]) {
  return make_uint2(a[0] | unsigned(a[1]) << 16, a[2] | unsigned(a[3]) << 16);
}

// One head per Head/4 adjacent lanes, 4 adjacent channels per lane (one rmsnorm_cast group). The rotation partner
// (channel +- Head/2) is lane XOR Head/8 in the same head. rmsnorm_cast squares each group in channel order, then
// reduces the Head/4 group sums by an XOR butterfly with shifts Head/8, ..., 1.
// All lanes, including those of masked tail heads, take part in every shuffle.
template <int Head>
__global__ void __launch_bounds__(kThreads)
    rope_rms_kernel(const unsigned short* __restrict__ q, int64_t q_ld, const unsigned short* __restrict__ k,
                    int64_t k_ld, const float* __restrict__ inv_freq, unsigned short* __restrict__ q_out,
                    unsigned short* __restrict__ k_out, int64_t heads, int64_t seq) {
  constexpr int kLanes = Head / 4, kHalf = Head / 2;
  const int lane = threadIdx.x % kLanes, channel = 4 * lane;
  const int64_t head = int64_t(blockIdx.x) * (kThreads / kLanes) + threadIdx.x / kLanes;
  const bool valid = head < heads;
  const int64_t token = head / kHeads, offset = (head % kHeads) * Head + channel;
  const float position = float(token % seq);
  const uint2 qv = valid ? *reinterpret_cast<const uint2*>(q + token * q_ld + offset) : make_uint2(0, 0);
  const uint2 kv = valid ? *reinterpret_cast<const uint2*>(k + token * k_ld + offset) : make_uint2(0, 0);
  const uint2 qp = make_uint2(__shfl_xor_sync(~0u, qv.x, kLanes / 2), __shfl_xor_sync(~0u, qv.y, kLanes / 2));
  const uint2 kp = make_uint2(__shfl_xor_sync(~0u, kv.x, kLanes / 2), __shfl_xor_sync(~0u, kv.y, kLanes / 2));
  unsigned short rq[4], rk[4];
  float sum_q = 0.f, sum_k = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float s, c;
    trig(mul(position, inv_freq[(channel + i) % kHalf]), s, c);
    float pq = mul(element(qp, i), s), pk = mul(element(kp, i), s);
    if (channel < kHalf) pq = -pq, pk = -pk;
    rq[i] = f32_to_bf16(fma_rn(element(qv, i), c, pq));
    rk[i] = f32_to_bf16(fma_rn(element(kv, i), c, pk));
    sum_q = square_add(rq[i], sum_q);
    sum_k = square_add(rk[i], sum_k);
  }
#pragma unroll
  for (int shift = kLanes / 2; shift >= 1; shift >>= 1) {
    sum_q = add(sum_q, __shfl_xor_sync(~0u, sum_q, shift, kLanes));
    sum_k = add(sum_k, __shfl_xor_sync(~0u, sum_k, shift, kLanes));
  }
  const float fq = inv_rms<Head>(sum_q), fk = inv_rms<Head>(sum_k);
  unsigned short oq[4], ok[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    oq[i] = f32_to_bf16(mul(bf16_to_f32(rq[i]), fq));
    ok[i] = f32_to_bf16(mul(bf16_to_f32(rk[i]), fk));
  }
  if (valid) {
    *reinterpret_cast<uint2*>(q_out + head * Head + channel) = pack(oq);
    *reinterpret_cast<uint2*>(k_out + head * Head + channel) = pack(ok);
  }
}

unsigned short* u16(at::Tensor const& t) { return reinterpret_cast<unsigned short*>(t.data_ptr()); }

template <int Head>
void launch(at::Tensor const& q, at::Tensor const& k, at::Tensor const& inv_freq, at::Tensor& q_out,
            at::Tensor& k_out) {
  const int64_t heads = q.size(0) * q.size(1) * kHeads, per_block = kThreads / (Head / 4);
  rope_rms_kernel<Head><<<(heads + per_block - 1) / per_block, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      u16(q), q.stride(1), u16(k), k.stride(1), inv_freq.data_ptr<float>(), u16(q_out), u16(k_out), heads, q.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_inv_freq(at::Tensor const& inv_freq, int64_t head) {
  TORCH_CHECK(inv_freq.is_cuda() && inv_freq.scalar_type() == at::kFloat && inv_freq.is_contiguous() &&
                  inv_freq.numel() == head / 2,
              "rope_rms: inv_freq must be a contiguous CUDA fp32 [head / 2] tensor");
}

// The Python supports() decides eligibility (including overlap); these checks only keep a bad call from touching
// memory.
std::vector<at::Tensor> run(at::Tensor const& q, at::Tensor const& k, at::Tensor const& inv_freq) {
  const int64_t head = q.size(-1);
  for (auto const* t : {&q, &k})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->dim() == 4 && t->sizes() == q.sizes() &&
                    t->size(2) == kHeads && t->stride(3) == 1 && t->stride(2) == head &&
                    t->stride(0) == t->size(1) * t->stride(1) && t->stride(1) % 4 == 0 &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 8 == 0,
                "rope_rms: q and k must be CUDA bf16 [C, S, 4, head] of one shape with strides (S * ld, ld, head, 1)");
  check_inv_freq(inv_freq, head);
  TORCH_CHECK(q.numel() > 0 && q.size(0) * q.size(1) * kHeads <= int64_t(INT_MAX) * 8,
              "rope_rms: q must be nonempty with few enough heads for an int32 grid");
  c10::cuda::CUDAGuard guard(q.device());
  auto q_out = at::empty(q.sizes(), q.options().memory_format(at::MemoryFormat::Contiguous));
  auto k_out = at::empty_like(q_out);
  if (head == 64)
    launch<64>(q, k, inv_freq, q_out, k_out);
  else if (head == 32)
    launch<32>(q, k, inv_freq, q_out, k_out);
  else
    TORCH_CHECK(false, "rope_rms: head must be 32 or 64, got ", head);
  return {q_out, k_out};
}

// Q and K of different lengths (the last row layer's 4 readout queries): the same per-head math as
// rope_rms_kernel on one tensor; blocks below q_blocks take Q, the rest K.
template <int Head>
__global__ void __launch_bounds__(kThreads)
    rope_rms_uneven_kernel(const unsigned short* __restrict__ q, int64_t q_ld, int64_t q_seq, int64_t q_heads,
                           unsigned short* __restrict__ q_out, const unsigned short* __restrict__ k, int64_t k_ld,
                           int64_t k_seq, int64_t k_heads, unsigned short* __restrict__ k_out,
                           const float* __restrict__ inv_freq, int64_t q_blocks) {
  constexpr int kLanes = Head / 4, kHalf = Head / 2;
  const bool is_k = blockIdx.x >= q_blocks;
  const unsigned short* x = is_k ? k : q;
  unsigned short* out = is_k ? k_out : q_out;
  const int64_t ld = is_k ? k_ld : q_ld, seq = is_k ? k_seq : q_seq, heads = is_k ? k_heads : q_heads;
  const int lane = threadIdx.x % kLanes, channel = 4 * lane;
  const int64_t head = (blockIdx.x - (is_k ? q_blocks : 0)) * (kThreads / kLanes) + threadIdx.x / kLanes;
  const bool valid = head < heads;
  const int64_t token = head / kHeads, offset = (head % kHeads) * Head + channel;
  const float position = float(token % seq);
  const uint2 v = valid ? *reinterpret_cast<const uint2*>(x + token * ld + offset) : make_uint2(0, 0);
  const uint2 p = make_uint2(__shfl_xor_sync(~0u, v.x, kLanes / 2), __shfl_xor_sync(~0u, v.y, kLanes / 2));
  unsigned short r[4];
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float s, c;
    trig(mul(position, inv_freq[(channel + i) % kHalf]), s, c);
    float rotated = mul(element(p, i), s);
    if (channel < kHalf) rotated = -rotated;
    r[i] = f32_to_bf16(fma_rn(element(v, i), c, rotated));
    sum = square_add(r[i], sum);
  }
#pragma unroll
  for (int shift = kLanes / 2; shift >= 1; shift >>= 1) sum = add(sum, __shfl_xor_sync(~0u, sum, shift, kLanes));
  const float f = inv_rms<Head>(sum);
  unsigned short o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = f32_to_bf16(mul(bf16_to_f32(r[i]), f));
  if (valid) *reinterpret_cast<uint2*>(out + head * Head + channel) = pack(o);
}

std::vector<at::Tensor> run_uneven(at::Tensor const& q, at::Tensor const& k, at::Tensor const& inv_freq) {
  const int64_t head = q.size(-1);
  for (auto const* t : {&q, &k})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->dim() == 4 && t->size(0) == q.size(0) &&
                    t->size(2) == kHeads && t->size(3) == head && t->stride(3) == 1 && t->stride(2) == head &&
                    t->stride(0) == t->size(1) * t->stride(1) && t->stride(1) % 4 == 0 && t->numel() > 0 &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 8 == 0,
                "rope_rms: q and k must be nonempty CUDA bf16 [C, S, 4, head] with strides (S * ld, ld, head, 1)");
  check_inv_freq(inv_freq, head);
  c10::cuda::CUDAGuard guard(q.device());
  auto q_out = at::empty(q.sizes(), q.options().memory_format(at::MemoryFormat::Contiguous));
  auto k_out = at::empty(k.sizes(), k.options().memory_format(at::MemoryFormat::Contiguous));
  auto go = [&](auto kernel, int64_t per_block) {
    const int64_t q_heads = q.size(0) * q.size(1) * kHeads, k_heads = k.size(0) * k.size(1) * kHeads;
    const int64_t q_blocks = (q_heads + per_block - 1) / per_block, k_blocks = (k_heads + per_block - 1) / per_block;
    TORCH_CHECK(q_blocks + k_blocks <= INT_MAX, "rope_rms: q and k have too many heads for an int32 grid");
    kernel<<<q_blocks + k_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        u16(q), q.stride(1), q.size(1), q_heads, u16(q_out), u16(k), k.stride(1), k.size(1), k_heads, u16(k_out),
        inv_freq.data_ptr<float>(), q_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  if (head == 64)
    go(rope_rms_uneven_kernel<64>, kThreads / 16);
  else if (head == 32)
    go(rope_rms_uneven_kernel<32>, kThreads / 8);
  else
    TORCH_CHECK(false, "rope_rms: head must be 32 or 64, got ", head);
  return {q_out, k_out};
}
}  // namespace rope_rms

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &rope_rms::run);
  m.def("run_uneven", &rope_rms::run_uneven);
}
