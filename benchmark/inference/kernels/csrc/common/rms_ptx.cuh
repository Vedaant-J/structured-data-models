// The FP32 steps of SDM's Triton rmsnorm_cast at channels = Head, as Triton lowers them to PTX.
#pragma once

namespace rms_ptx {
__device__ __forceinline__ float add(float a, float b) {
  float v;
  asm("add.rn.f32 %0,%1,%2;" : "=f"(v) : "f"(a), "f"(b));
  return v;
}
__device__ __forceinline__ float mul(float a, float b) {
  float v;
  asm("mul.rn.f32 %0,%1,%2;" : "=f"(v) : "f"(a), "f"(b));
  return v;
}
// sum + x * x for the BF16 x with bits `x`, one rounding.
__device__ __forceinline__ float square_add(unsigned short x, float sum) {
  float v;
  asm("fma.rn.f32.bf16 %0,%1,%1,%2;" : "=f"(v) : "h"(x), "f"(sum));
  return v;
}
// rsqrt(sum / Head + 1e-6) exactly as Triton emits it: div.full, add.rn, rsqrt.approx.ftz.
template <int Head>
__device__ __forceinline__ float inv_rms(float sum) {
  float mean, shifted, v;
  asm("div.full.f32 %0,%1,%2;" : "=f"(mean) : "f"(sum), "f"(float(Head)));
  asm("add.rn.f32 %0,%1,0f358637BD;" : "=f"(shifted) : "f"(mean));
  asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(v) : "f"(shifted));
  return v;
}
}  // namespace rms_ptx
