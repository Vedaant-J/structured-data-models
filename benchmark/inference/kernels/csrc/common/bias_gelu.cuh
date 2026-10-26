// cuBLAS addmm's epilogue (bias added in FP32, RN to BF16) followed by Inductor's Triton GELU on the BF16 value.
#pragma once
#include <cutlass/epilogue/thread/linear_combination.h>
#include "common/triton_gelu.cuh"

namespace bias_gelu {
using Linear = cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>;

// Each module derives its own named op from Op, so its kernel symbol names the module.
struct Op : Linear {
  using Linear::Linear;
  CUTLASS_DEVICE static FragmentOutput gelu(FragmentOutput x) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      const cutlass::bfloat16_t v = x[i];
      x[i] = cutlass::bfloat16_t::bitcast(__bfloat16_as_ushort(triton_gelu_bf16(__ushort_as_bfloat16(v.raw()))));
    }
    return x;
  }
  CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc, FragmentSource const& bias) const {
    return gelu(Linear::operator()(acc, bias));
  }
  CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc) const {
    return gelu(Linear::operator()(acc));
  }
};
}  // namespace bias_gelu
