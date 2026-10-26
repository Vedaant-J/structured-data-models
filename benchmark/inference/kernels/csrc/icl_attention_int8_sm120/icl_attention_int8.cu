// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// 8-bit self-attention for SM120, head dim 64, no mask.
//
// prep: K - mean_s(K) quantized to INT8 per 64-key block; V quantized to FP8 E4M3 per channel and stored
// transposed [D, S] with keys permuted inside every 32-key group to match the P fragment layout.
// attend: Q quantized to INT8 per row in the prologue, S = Q K^T on the INT8 MMA, online softmax in FP32,
// P (scaled by 448) as FP8 times V on the FP8 MMA with FP32 accumulation.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace icl_attention_int8 {

constexpr int D = 64;
constexpr int BN = 64;
constexpr float kLog2P = 8.807354922057604f;  // log2(448)
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kMagic = 0x4B400000;  // 1.5 * 2^23
constexpr float kMagicF = 12582912.f;

struct Strided {  // bf16 or fp16 bits
  const uint16_t* p;
  int64_t sb, ss, sh;  // batch, token, head strides (elements)
  __device__ const uint16_t* at(int b, int s, int h) const { return p + b * sb + s * ss + h * sh; }
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// d = a b + {c, c, c, c}
__device__ __forceinline__ void mma_s8_init(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int c) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}
__device__ __forceinline__ void mma_e4m3(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// byte i of the result = x_i
__device__ __forceinline__ uint32_t pack_e4m3(float x0, float x1, float x2, float x3) {
  uint32_t r;
  asm("{ .reg .b16 l, h;\n"
      "cvt.rn.satfinite.e4m3x2.f32 l, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32 h, %4, %3;\n"
      "mov.b32 %0, {l, h}; }"
      : "=r"(r)
      : "f"(x0), "f"(x1), "f"(x2), "f"(x3));
  return r;
}
template <bool F16>
__device__ __forceinline__ float tof(uint16_t x) {
  if constexpr (F16)
    return __half2float(__ushort_as_half(x));
  else
    return __bfloat162float(__ushort_as_bfloat16(x));
}
template <bool F16>
__device__ __forceinline__ uint32_t pack16(float lo, float hi) {
  if constexpr (F16) {
    __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
  } else {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
  }
}
__device__ __forceinline__ int quant8(float x) { return max(-127, min(127, __float2int_rn(x))); }
__device__ __forceinline__ uint32_t pack_s8(int a, int b, int c, int d) {
  return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((uint32_t)(d & 0xff) << 24);
}

// 64-byte rows, 16-byte chunks XOR-swizzled so ldmatrix rows hit distinct banks.
__device__ __forceinline__ int swz64(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

// Logical position p in a 32-key group holds key sigma(p), so that a thread's
// QK^T accumulators are its P A-fragment bytes in order.
__device__ __forceinline__ int sigma(int p) {
  int h = p >> 4, t = (p >> 2) & 3, i = p & 3;
  return 16 * h + 2 * t + (i & 1) + 8 * (i >> 1);
}

// Per (b, h, chunk): kpart[d] = sum_s K over the chunk, vmax[d] = max_s |V| (as uint bits).
constexpr int kStatTokens = 1024;
template <bool F16>
__global__ void __launch_bounds__(256) stats_kernel(Strided k, Strided v, int S, int H, float* kpart, unsigned* vmax) {
  int b = blockIdx.z, h = blockIdx.y, s0 = blockIdx.x * kStatTokens;
  int lane8 = threadIdx.x & 7, row = threadIdx.x >> 3;  // 32 token rows per pass
  float sum[8] = {}, mx[8] = {};
  int s1 = min(S, s0 + kStatTokens);
  for (int s = s0 + row; s < s1; s += 32) {
    uint4 ku = *reinterpret_cast<const uint4*>(k.at(b, s, h) + lane8 * 8);
    const uint16_t* ke = reinterpret_cast<const uint16_t*>(&ku);
#pragma unroll
    for (int i = 0; i < 8; ++i) sum[i] += tof<F16>(ke[i]);
    uint4 vu = *reinterpret_cast<const uint4*>(v.at(b, s, h) + lane8 * 8);
    const uint16_t* ve = reinterpret_cast<const uint16_t*>(&vu);
#pragma unroll
    for (int i = 0; i < 8; ++i) mx[i] = fmaxf(mx[i], fabsf(tof<F16>(ve[i])));
  }
  __shared__ float red[2][32][65];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    red[0][row][lane8 * 8 + i] = sum[i];
    red[1][row][lane8 * 8 + i] = mx[i];
  }
  __syncthreads();
  if (threadIdx.x < 64) {
    float s = 0.f, m = 0.f;
    for (int r = 0; r < 32; ++r) {
      s += red[0][r][threadIdx.x];
      m = fmaxf(m, red[1][r][threadIdx.x]);
    }
    kpart[((int64_t)(b * H + h) * gridDim.x + blockIdx.x) * D + threadIdx.x] = s;
    atomicMax(vmax + (b * H + h) * D + threadIdx.x, __float_as_uint(m));
  }
}

// Fixed-order sum of the chunk partials, so the K mean is bitwise reproducible.
__global__ void __launch_bounds__(D) ksum_kernel(const float* kpart, int chunks, float* ksum) {
  const float* src = kpart + (int64_t)blockIdx.x * chunks * D + threadIdx.x;
  float s = 0.f;
  for (int c = 0; c < chunks; ++c) s += src[c * D];
  ksum[blockIdx.x * D + threadIdx.x] = s;
}

// One block per (64-key block, h, b), 256 threads.
template <bool F16>
__global__ void __launch_bounds__(256) quant_kernel(Strided k, Strided v, int S, int S_pad, int H, const float* ksum,
                                                    const unsigned* vmax, int8_t* kq, float* kscale, uint8_t* vt) {
  int b = blockIdx.z, h = blockIdx.y, blk = blockIdx.x, s0 = blk * BN;
  int bh = b * H + h;
  __shared__ float vs[BN][D + 1];
  __shared__ float wmax[8];
  int lane8 = threadIdx.x & 7, row = threadIdx.x >> 3;  // rows row, row + 32
  float kv[2][8];
  float amax = 0.f, tmax[2] = {0.f, 0.f};
  float inv_n = 1.f / S;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int s = s0 + row + 32 * r;
#pragma unroll
    for (int i = 0; i < 8; ++i) kv[r][i] = 0.f;
    if (s < S) {
      uint4 u = *reinterpret_cast<const uint4*>(k.at(b, s, h) + lane8 * 8);
      const uint16_t* e = reinterpret_cast<const uint16_t*>(&u);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float x = tof<F16>(e[i]);
        x -= ksum[bh * D + lane8 * 8 + i] * inv_n;
        kv[r][i] = x;
        tmax[r] = fmaxf(tmax[r], fabsf(x));
      }
    }
#pragma unroll
    for (int o = 4; o; o >>= 1) tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], o));
    amax = fmaxf(amax, tmax[r]);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = amax;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int s = s0 + row + 32 * r;
    float f[8] = {};
    if (s < S) {
      uint4 u = *reinterpret_cast<const uint4*>(v.at(b, s, h) + lane8 * 8);
      const uint16_t* e = reinterpret_cast<const uint16_t*>(&u);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float m = __uint_as_float(vmax[bh * D + lane8 * 8 + i]);
        f[i] = m > 0.f ? tof<F16>(e[i]) * (448.f / m) : 0.f;
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) vs[row + 32 * r][lane8 * 8 + i] = f[i];
  }
  __syncthreads();
  amax = 0.f;
#pragma unroll
  for (int w = 0; w < 8; ++w) amax = fmaxf(amax, wmax[w]);
  if (threadIdx.x == 0) kscale[bh * (S_pad / BN) + blk] = amax / 127.f;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int s = s0 + row + 32 * r;
    float inv = amax > 0.f ? 127.f / amax : 0.f;
    uint2 out;
    out.x = pack_s8(quant8(kv[r][0] * inv), quant8(kv[r][1] * inv), quant8(kv[r][2] * inv), quant8(kv[r][3] * inv));
    out.y = pack_s8(quant8(kv[r][4] * inv), quant8(kv[r][5] * inv), quant8(kv[r][6] * inv), quant8(kv[r][7] * inv));
    *reinterpret_cast<uint2*>(kq + ((int64_t)bh * S_pad + s) * D + lane8 * 8) = out;
  }
  // 64 channels x 64 keys; each thread writes 16 consecutive logical keys of one channel.
  int c = threadIdx.x >> 2, q = threadIdx.x & 3;
  float x[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int p = q * 16 + i;
    x[i] = vs[(p & ~31) + sigma(p & 31)][c];
  }
  uint4 o;
  o.x = pack_e4m3(x[0], x[1], x[2], x[3]);
  o.y = pack_e4m3(x[4], x[5], x[6], x[7]);
  o.z = pack_e4m3(x[8], x[9], x[10], x[11]);
  o.w = pack_e4m3(x[12], x[13], x[14], x[15]);
  *reinterpret_cast<uint4*>(vt + ((int64_t)bh * D + c) * S_pad + s0 + q * 16) = o;
}

// WM m16 tiles of query rows per warp, NW warps, a STAGES-deep cp.async ring of 64-key K/V tiles.
constexpr int WM = 2, NW = 4, STAGES = 3;
constexpr int BM = 16 * WM * NW;
constexpr int NT = NW * 32;
constexpr int KBYTES = BN * D, VBYTES = BN * D;
constexpr int STAGE = KBYTES + VBYTES;
constexpr int SMEM = STAGES * STAGE;
constexpr int NK = KBYTES / 16 / NT, NV = VBYTES / 16 / NT;

template <bool F16>
__global__ void __launch_bounds__(NT, 1)
    attend_kernel(Strided q, const int8_t* __restrict__ kq, const float* __restrict__ kscale,
                  const uint8_t* __restrict__ vt, const unsigned* __restrict__ vmax, uint16_t* out, int64_t osb,
                  int64_t oss, int64_t osh, int S, int S_pad, int H, float sm_log2) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int b = blockIdx.z, h = blockIdx.y, bh = b * H + h;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.x * BM + warp * 16 * WM;
  const int ntiles = S_pad / BN;
  const float* ks = kscale + (int64_t)bh * ntiles;
  const uint32_t sbase = smem_u32(smem);

  const int8_t* kg = kq + (int64_t)bh * S_pad * D + tid * 16;
  const uint8_t* vg = vt + ((int64_t)bh * D + (tid >> 2)) * S_pad + (tid & 3) * 16;
  const int64_t vstep = (int64_t)(NT / 4) * S_pad;
  const uint32_t sw = swz64(tid >> 2, tid & 3);
  auto load = [&](int tile, int stage) {
    const uint32_t st = sbase + stage * STAGE;
#pragma unroll
    for (int i = 0; i < NK; ++i) cp_async16(st + sw + i * NT * 16, kg + (int64_t)tile * BN * D + i * NT * 16);
#pragma unroll
    for (int i = 0; i < NV; ++i) cp_async16(st + KBYTES + sw + i * NT * 16, vg + i * vstep + tile * BN);
  };

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < ntiles) load(i, i);
    cp_commit();
  }

  // Q prologue: per-row INT8 quantization straight into A fragments.
  uint32_t qa[WM][2][4];
  float qs[WM][2];
#pragma unroll
  for (int i = 0; i < WM; ++i) {
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      int row = m0 + 16 * i + g + 8 * hr;
      float x[16];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint2 u = make_uint2(0, 0);
        if (row < S) u = *reinterpret_cast<const uint2*>(q.at(b, row, h) + 16 * j + 4 * t);
        const uint16_t* e = reinterpret_cast<const uint16_t*>(&u);
#pragma unroll
        for (int e4 = 0; e4 < 4; ++e4) x[4 * j + e4] = tof<F16>(e[e4]);
      }
      float amax = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(x[j]));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
      float inv = amax > 0.f ? 127.f / amax : 0.f;
      qs[i][hr] = amax / 127.f * sm_log2;
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float* y = x + 4 * (2 * s + half);
          qa[i][s][2 * half + hr] =
              pack_s8(quant8(y[0] * inv), quant8(y[1] * inv), quant8(y[2] * inv), quant8(y[3] * inv));
        }
    }
  }

  float o[WM][8][4];
  float mrow[WM][2], lrow[WM][2];
#pragma unroll
  for (int i = 0; i < WM; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[i][j][e] = 0.f;
    mrow[i][0] = mrow[i][1] = -INFINITY;
    lrow[i][0] = lrow[i][1] = 0.f;
  }

  const int lrow8 = lane & 7, lmat = lane >> 3;
  float ksc = ks[0];
  int rd = 0, wr = STAGES - 1;

  auto step = [&](int tile, auto masked_t) {
    constexpr bool masked = decltype(masked_t)::value;
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (tile + STAGES - 1 < ntiles) load(tile + STAGES - 1, wr);
    cp_commit();
    wr = wr == STAGES - 1 ? 0 : wr + 1;
    const float ksn = tile + 1 < ntiles ? ks[tile + 1] : 0.f;
    const uint32_t kst = sbase + rd * STAGE, vst = kst + KBYTES;

    // Scores land as the bits of 1.5 * 2^23 + s (a float, exact for |s| < 2^22).
    int sc[WM][8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t kb[4];
      ldsm_x4(kb, kst + swz64(8 * j + lrow8, lmat));
#pragma unroll
      for (int i = 0; i < WM; ++i) {
        mma_s8_init(sc[i][j], qa[i][0], kb[0], kb[1], kMagic);
        mma_s8(sc[i][j], qa[i][1], kb[2], kb[3]);
      }
    }

    if (masked) {
#pragma unroll
      for (int i = 0; i < WM; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            if (tile * BN + 8 * j + 2 * t + (e & 1) >= S) sc[i][j][e] = 0x4B000000;  // s = -2^22
    }

    float cs[WM][2], mnew[WM][2];
    bool grow = false;
#pragma unroll
    for (int i = 0; i < WM; ++i) {
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        int mx = sc[i][0][2 * hr];
#pragma unroll
        for (int j = 0; j < 8; ++j) mx = max(mx, max(sc[i][j][2 * hr], sc[i][j][2 * hr + 1]));
        mx = max(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = max(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        const float mxf = __int_as_float(mx) - kMagicF;
        cs[i][hr] = qs[i][hr] * ksc;
        mnew[i][hr] = fmaxf(mrow[i][hr], mxf * cs[i][hr]);
        grow |= mnew[i][hr] > mrow[i][hr];
      }
    }
    if (__any_sync(0xffffffff, grow)) {
#pragma unroll
      for (int i = 0; i < WM; ++i)
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
          const float alpha = ex2(mrow[i][hr] - mnew[i][hr]);
          lrow[i][hr] *= alpha;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            o[i][j][2 * hr] *= alpha;
            o[i][j][2 * hr + 1] *= alpha;
          }
          mrow[i][hr] = mnew[i][hr];
        }
    }
    uint32_t pa[WM][2][4];
#pragma unroll
    for (int i = 0; i < WM; ++i) {
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const float bias = kLog2P - mrow[i][hr] - kMagicF * cs[i][hr];
        float p[8][2];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float x = fmaf(__int_as_float(sc[i][j][2 * hr + e]), cs[i][hr], bias);
            float z = ex2(x);
            if (masked && tile * BN + 8 * j + 2 * t + e >= S) z = 0.f;
            p[j][e] = z;
          }
        float s4[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) s4[u] = (p[2 * u][0] + p[2 * u][1]) + (p[2 * u + 1][0] + p[2 * u + 1][1]);
        lrow[i][hr] += (s4[0] + s4[1]) + (s4[2] + s4[3]);
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          pa[i][s][hr] = pack_e4m3(p[4 * s][0], p[4 * s][1], p[4 * s + 1][0], p[4 * s + 1][1]);
          pa[i][s][2 + hr] = pack_e4m3(p[4 * s + 2][0], p[4 * s + 2][1], p[4 * s + 3][0], p[4 * s + 3][1]);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t vb[4];
      ldsm_x4(vb, vst + swz64(8 * j + lrow8, lmat));
#pragma unroll
      for (int i = 0; i < WM; ++i) {
        mma_e4m3(o[i][j], pa[i][0], vb[0], vb[1]);
        mma_e4m3(o[i][j], pa[i][1], vb[2], vb[3]);
      }
    }
    ksc = ksn;
    rd = rd == STAGES - 1 ? 0 : rd + 1;
  };
  for (int tile = 0; tile < ntiles - 1; ++tile) step(tile, std::false_type{});
  if (S % BN)
    step(ntiles - 1, std::true_type{});
  else
    step(ntiles - 1, std::false_type{});
  cp_wait<0>();

#pragma unroll
  for (int i = 0; i < WM; ++i) {
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      float l = lrow[i][hr];
      l += __shfl_xor_sync(0xffffffff, l, 1);
      l += __shfl_xor_sync(0xffffffff, l, 2);
      const float inv = 1.f / l;
      int row = m0 + 16 * i + g + 8 * hr;
      if (row >= S) continue;
      uint16_t* dst = out + b * osb + row * oss + h * osh;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int d = 8 * j + 2 * t;
        float f0 = inv, f1 = inv;
        f0 *= __uint_as_float(vmax[bh * D + d]) * (1.f / 448.f);
        f1 *= __uint_as_float(vmax[bh * D + d + 1]) * (1.f / 448.f);
        *reinterpret_cast<uint32_t*>(dst + d) = pack16<F16>(o[i][j][2 * hr] * f0, o[i][j][2 * hr + 1] * f1);
      }
    }
  }
}

Strided strided(const at::Tensor& x) {  // x: [B, H, S, D] logical
  return {reinterpret_cast<const uint16_t*>(x.data_ptr()), x.stride(0), x.stride(2), x.stride(1)};
}

void check(const at::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) && x.dim() == 4 &&
                  x.size(3) == D && x.stride(3) == 1,
              "icl_attention_int8: ", name, " must be a CUDA bf16 or fp16 [B, H, S, 64] tensor with unit last stride");
  TORCH_CHECK(x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0 && x.stride(2) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "icl_attention_int8: ", name, " rows must be 16-byte aligned");
}

// q, k, v, out: [B, H, S, 64] views of one dtype, any strides with a unit last stride.
void run(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, double sm_scale) {
  check(q, "q");
  check(k, "k");
  check(v, "v");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "icl_attention_int8: k and v must have the dtype of q");
  TORCH_CHECK(out.scalar_type() == q.scalar_type() && out.sizes() == q.sizes() && out.stride(3) == 1 &&
                  out.stride(0) % 2 == 0 && out.stride(1) % 2 == 0 && out.stride(2) % 2 == 0,
              "icl_attention_int8: out must have the dtype and shape of q, a unit last stride and even strides");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "icl_attention_int8: k and v must have the shape of q");
  int B = q.size(0), H = q.size(1), S = q.size(2);
  TORCH_CHECK(S > 0 && (int64_t)S * std::max(q.stride(2), std::max(k.stride(2), v.stride(2))) < (1ll << 31) &&
                  (int64_t)B * H * S * D < (1ll << 31),
              "icl_attention_int8: S must be positive and element offsets must fit in int32");
  c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int S_pad = (S + BN - 1) / BN * BN;
  auto f32 = q.options().dtype(at::kFloat);
  auto ws = at::zeros({2, B * H * D}, f32);
  float* ksum = ws.data_ptr<float>();
  unsigned* vmax = reinterpret_cast<unsigned*>(ksum + B * H * D);
  auto kq = at::empty({B, H, S_pad, D}, q.options().dtype(at::kChar));
  auto kscale = at::empty({B, H, S_pad / BN}, f32);
  auto vt = at::empty({B, H, D, S_pad}, q.options().dtype(at::kByte));
  const int chunks = (S + kStatTokens - 1) / kStatTokens;
  auto kpart = at::empty({B * H, chunks, D}, f32);
  auto launch = [&](auto f16_t) {
    constexpr bool F16 = decltype(f16_t)::value;
    stats_kernel<F16>
        <<<dim3(chunks, H, B), 256, 0, stream>>>(strided(k), strided(v), S, H, kpart.data_ptr<float>(), vmax);
    ksum_kernel<<<B * H, D, 0, stream>>>(kpart.data_ptr<float>(), chunks, ksum);
    quant_kernel<F16><<<dim3(S_pad / BN, H, B), 256, 0, stream>>>(strided(k), strided(v), S, S_pad, H, ksum, vmax,
                                                                  kq.data_ptr<int8_t>(), kscale.data_ptr<float>(),
                                                                  vt.data_ptr<uint8_t>());
    attend_kernel<F16><<<dim3((S + BM - 1) / BM, H, B), NT, SMEM, stream>>>(
        strided(q), kq.data_ptr<int8_t>(), kscale.data_ptr<float>(), vt.data_ptr<uint8_t>(), vmax,
        reinterpret_cast<uint16_t*>(out.data_ptr()), out.stride(0), out.stride(2), out.stride(1), S, S_pad, H,
        float(sm_scale * kLog2e));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  if (q.scalar_type() == at::kHalf)
    launch(std::true_type{});
  else
    launch(std::false_type{});
}

}  // namespace icl_attention_int8

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &icl_attention_int8::run); }
