#pragma once
#include <cuda_bf16.h>

// Inductor's Triton GELU (erf form) as lowered to PTX on CUDA 13: same constants, same op order,
// same FTZ flags. Not the toolkit erff.
namespace triton_gelu_ptx {
__device__ __forceinline__ float bits(unsigned x) { return __uint_as_float(x); }
__device__ __forceinline__ float mul(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float add(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float sub(float a, float b) {
  float r;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float abs_ftz(float a) {
  float r;
  asm("abs.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float r;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float exp2_ftz(float a) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float erf_observed(float s) {
  const float a = abs_ftz(s);
  const bool large = a >= bits(0x3f8060feu);
  const float t = large ? a : mul(s, s);
  float p = bits(large ? 0x3789ca3cu : 0x38b1e96au);
  p = fma_ftz(p, t, bits(large ? 0xb9f560b9u : 0xba574d20u));
  p = fma_ftz(p, t, bits(large ? 0x3bac840bu : 0x3baad5eau));
  p = fma_ftz(p, t, bits(large ? 0xbd0c8162u : 0xbcdc1be7u));
  p = fma_ftz(p, t, bits(large ? 0x3e1cf906u : 0x3de718afu));
  p = fma_ftz(p, t, bits(large ? 0x3f6a937eu : 0xbec093acu));
  p = fma_ftz(p, t, bits(large ? 0x3f20d842u : 0x3e0375d3u));
  const float u = large ? -t : s;
  const float v = fma_ftz(p, u, u);
  if (!large) return v;
  const float r = sub(1.0f, exp2_ftz(v));
  return bits(__float_as_uint(r) | (__float_as_uint(s) & 0x80000000u));
}
}  // namespace triton_gelu_ptx

__device__ __forceinline__ __nv_bfloat16 triton_gelu_bf16(__nv_bfloat16 input) {
  using namespace triton_gelu_ptx;
  const float z = __bfloat162float(input);
  const float s = mul(z, bits(0x3f3504f3u));
  const float e = erf_observed(s);
  return __float2bfloat16_rn(mul(mul(z, 0.5f), add(e, 1.0f)));
}
