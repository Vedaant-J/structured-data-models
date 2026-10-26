#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace icl_attention {

constexpr int D = 64, BN = 64, TILE = BN * D;

struct Params {
  CUtensorMap tq, tk, tv;
  __nv_bfloat16* o;
  float* ws;
  int* counts;
  int64_t ob, os, oh;
  int sq, skv, heads, nblk, tiles, whole, splits;
  float scale_log2;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Rows of 64 bf16 = 8 chunks of 16 B, chunk index XOR (row & 7): the TMA
// 128B swizzle, conflict-free for ldmatrix.
__device__ __forceinline__ int swz(int row, int chunk) { return row * D + ((chunk ^ (row & 7)) << 3); }

__device__ __forceinline__ void bar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void bar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void bar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], "
      "%1;\n@!p bra W;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bar_expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
// Rows [row, row + box) of head h, batch b -> swizzled smem tile.
__device__ __forceinline__ void tma(uint32_t dst, const CUtensorMap* map, int row, int h, int b, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::"
      "complete_tx::bytes [%0], [%1, {0, %2, %3, %4}], [%5];\n" ::"r"(dst),
      "l"(map), "r"(h), "r"(row), "r"(b), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm4t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// WARPS warps, each owning MT m16 tiles of query rows; a STAGES-deep TMA
// ring of 64-key K/V tiles with full/empty mbarriers. CTAs below `whole` own
// a query block; the blocks after them (the last wave) are split into
// `splits` key ranges, merged by the last CTA to finish the block.
template <int WARPS, int MT, int STAGES>
struct Kernel {
  static constexpr int BM = WARPS * 16 * MT;
  static constexpr int THREADS = WARPS * 32;
  static constexpr int NT = BN / 8;
  static constexpr int SMEM_Q = BM * D * 2;
  static constexpr int SMEM = SMEM_Q + STAGES * 2 * TILE * 2 + 8 * (2 * STAGES + 1);
  static constexpr int PART = MT * 8 * 4 + MT * 4;  // floats per thread

  struct State {
    uint32_t qf[MT][4][4];
    float o[MT][8][4];
    float mx[MT][2], msc[MT][2], l[MT][2];
  };

  template <bool MASK>
  static __device__ __forceinline__ void step(State& st, uint32_t kt, uint32_t vt, int col0, int skv, float sl2) {
    const int lane = threadIdx.x & 31;
    const int rx = lane & 7;
    float s[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) s[mt][j][e] = 0.f;
    const uint32_t koff = (((lane >> 4) << 3) + rx) * D * 2;
    const int kx = (lane >> 3) & 1;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
#pragma unroll
      for (int jp = 0; jp < NT / 2; ++jp) {
        uint32_t bf[4];
        ldsm4(bf, kt + koff + jp * 16 * D * 2 + (((kk * 2 + kx) ^ rx) << 4));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma(s[mt][2 * jp], st.qf[mt][kk], bf[0], bf[1]);
          mma(s[mt][2 * jp + 1], st.qf[mt][kk], bf[2], bf[3]);
        }
      }

    if (MASK) {
      const int c = col0 + 2 * (lane & 3);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            if (c + j * 8 + (e & 1) >= skv) s[mt][j][e] = -INFINITY;
    }

    // Exact online softmax; the warp-wide max and O rescale only run when
    // some row of the warp has a new maximum.
    float tm[MT][2];
    bool grew = false;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float v[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) v[j] = fmaxf(s[mt][j][2 * r], s[mt][j][2 * r + 1]);
#pragma unroll
        for (int w = NT / 2; w > 0; w /= 2)
#pragma unroll
          for (int j = 0; j < w; ++j) v[j] = fmaxf(v[j], v[j + w]);
        tm[mt][r] = v[0];
        grew |= v[0] > st.mx[mt][r];
      }
    if (__any_sync(0xffffffff, grew)) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          float m = tm[mt][r];
          m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
          m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
          const float nm = fmaxf(st.mx[mt][r], m);
          const float f = ex2((st.mx[mt][r] - nm) * sl2);
          st.mx[mt][r] = nm;
          st.msc[mt][r] = nm * sl2;
          st.l[mt][r] *= f;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            st.o[mt][j][2 * r] *= f;
            st.o[mt][j][2 * r + 1] *= f;
          }
        }
    }

    uint32_t pf[MT][NT / 2][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        float e[4];
#pragma unroll
        for (int x = 0; x < 4; ++x) e[x] = ex2(fmaf(s[mt][j][x], sl2, -st.msc[mt][x >> 1]));
        st.l[mt][0] += e[0] + e[1];
        st.l[mt][1] += e[2] + e[3];
        pf[mt][j >> 1][(j & 1) * 2] = pack_bf16(e[0], e[1]);
        pf[mt][j >> 1][(j & 1) * 2 + 1] = pack_bf16(e[2], e[3]);
      }

    const uint32_t voff = (lane & 15) * D * 2;
    const int vx = lane >> 4;
#pragma unroll
    for (int kk = 0; kk < NT / 2; ++kk)
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t bf[4];
        ldsm4t(bf, vt + voff + kk * 16 * D * 2 + (((dp * 2 + vx) ^ rx) << 4));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma(st.o[mt][2 * dp], pf[mt][kk], bf[0], bf[1]);
          mma(st.o[mt][2 * dp + 1], pf[mt][kk], bf[2], bf[3]);
        }
      }
  }

  static __device__ __forceinline__ float* part(const Params& p, int slot) {
    return p.ws + ((int64_t)slot * THREADS + threadIdx.x) * PART;
  }

  static __device__ void store(const Params& p, State& st, int blk) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int mb = blk % p.nblk, bh = blk / p.nblk;
    const int h = bh % p.heads, b = bh / p.heads;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float s = st.l[mt][r];
        s += __shfl_xor_sync(0xffffffff, s, 1);
        s += __shfl_xor_sync(0xffffffff, s, 2);
        const float inv = 1.f / s;
        const int row = mb * BM + (warp * MT + mt) * 16 + (lane >> 2) + r * 8;
        if (row < p.sq) {
          __nv_bfloat16* out = p.o + b * p.ob + h * p.oh + row * p.os;
#pragma unroll
          for (int j = 0; j < 8; ++j)
            *reinterpret_cast<uint32_t*>(out + j * 8 + 2 * (lane & 3)) =
                pack_bf16(st.o[mt][j][2 * r] * inv, st.o[mt][j][2 * r + 1] * inv);
        }
      }
  }

  // Partial result of a split block: publish it; the last split to finish
  // merges all parts in split order and writes the output.
  static __device__ void merge(const Params& p, State& st, int blk, int split, float sl2) {
    __shared__ int last;
    const int first = (blk - p.whole) * p.splits;
    float4* w = reinterpret_cast<float4*>(part(p, first + split));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        w[mt * 8 + j] = make_float4(st.o[mt][j][0], st.o[mt][j][1], st.o[mt][j][2], st.o[mt][j][3]);
      w[MT * 8 + mt] = make_float4(st.mx[mt][0], st.mx[mt][1], st.l[mt][0], st.l[mt][1]);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(p.counts + blk, 1) == p.splits - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const float4* x0 = reinterpret_cast<const float4*>(part(p, first));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float4 v = __ldcg(x0 + mt * 8 + j);
        st.o[mt][j][0] = v.x, st.o[mt][j][1] = v.y;
        st.o[mt][j][2] = v.z, st.o[mt][j][3] = v.w;
      }
      const float4 ml = __ldcg(x0 + MT * 8 + mt);
      st.mx[mt][0] = ml.x, st.mx[mt][1] = ml.y;
      st.l[mt][0] = ml.z, st.l[mt][1] = ml.w;
    }
    for (int sp = 1; sp < p.splits; ++sp) {
      const float4* x = reinterpret_cast<const float4*>(part(p, first + sp));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float4 ml = __ldcg(x + MT * 8 + mt);
        float f[2], g[2];
        const float om[2] = {ml.x, ml.y}, ol[2] = {ml.z, ml.w};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float nm = fmaxf(st.mx[mt][r], om[r]);
          f[r] = ex2((st.mx[mt][r] - nm) * sl2);
          g[r] = ex2((om[r] - nm) * sl2);
          st.mx[mt][r] = nm;
          st.l[mt][r] = st.l[mt][r] * f[r] + ol[r] * g[r];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float4 v = __ldcg(x + mt * 8 + j);
          st.o[mt][j][0] = st.o[mt][j][0] * f[0] + v.x * g[0];
          st.o[mt][j][1] = st.o[mt][j][1] * f[0] + v.y * g[0];
          st.o[mt][j][2] = st.o[mt][j][2] * f[1] + v.z * g[1];
          st.o[mt][j][3] = st.o[mt][j][3] * f[1] + v.w * g[1];
        }
      }
    }
    store(p, st, blk);
  }

  static __device__ void run(const Params& p) {
    extern __shared__ __align__(1024) __nv_bfloat16 smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int blk = blockIdx.x, split = 0, t0 = 0, t1 = p.tiles;
    if (blk >= p.whole) {
      const int j = blk - p.whole;
      blk = p.whole + j / p.splits;
      split = j % p.splits;
      t0 = split * p.tiles / p.splits;
      t1 = (split + 1) * p.tiles / p.splits;
    }
    const int bh = blk / p.nblk, h = bh % p.heads, b = bh / p.heads;
    const uint32_t qs = smem_u32(smem), ks = qs + SMEM_Q;
    const uint32_t vs = ks + STAGES * TILE * 2, bars = vs + STAGES * TILE * 2;
    const uint32_t qbar = bars + 16 * STAGES;
    const float sl2 = p.scale_log2;
    const int n = t1 - t0;

    auto load = [&](int i) {
      if (i >= n) return;
      const int slot = i % STAGES;
      const uint32_t full = bars + 8 * slot;
      if (i >= STAGES) bar_wait(bars + 8 * (STAGES + slot), ((i / STAGES) - 1) & 1);
      bar_expect(full, 2 * TILE * 2);
      tma(ks + slot * TILE * 2, &p.tk, (t0 + i) * BN, h, b, full);
      tma(vs + slot * TILE * 2, &p.tv, (t0 + i) * BN, h, b, full);
    };
    if (tid == 0) {
      for (int s = 0; s < STAGES; ++s) {
        bar_init(bars + 8 * s, 1);
        bar_init(bars + 8 * (STAGES + s), WARPS);
      }
      bar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
      bar_expect(qbar, BM * D * 2);
      tma(qs, &p.tq, (blk % p.nblk) * BM, h, b, qbar);
      for (int s = 0; s < STAGES - 1; ++s) load(s);
    }
    __syncthreads();

    State st;
    bar_wait(qbar, 0);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const int row = (warp * MT + mt) * 16 + (lane & 15);
        ldsm4(st.qf[mt][kk], qs + 2 * swz(row, kk * 2 + (lane >> 4)));
      }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) st.o[mt][j][e] = 0.f;
      st.mx[mt][0] = st.mx[mt][1] = -INFINITY;
      st.l[mt][0] = st.l[mt][1] = 0.f;
    }

    auto tile = [&](int i, auto mask) {
      const int slot = i % STAGES;
      bar_wait(bars + 8 * slot, (i / STAGES) & 1);
      if (tid == 0) load(i + STAGES - 1);
      step<decltype(mask)::value>(st, ks + slot * TILE * 2, vs + slot * TILE * 2, (t0 + i) * BN, p.skv, sl2);
      __syncwarp();
      if (lane == 0) bar_arrive(bars + 8 * (STAGES + slot));
    };
    const int open = t1 == p.tiles ? n - 1 : n;
    for (int i = 0; i < open; ++i) tile(i, std::false_type{});
    if (open < n) tile(open, std::true_type{});
    if (blk < p.whole)
      store(p, st, blk);
    else
      merge(p, st, blk, split, sl2);
  }
};

template <int W, int MT, int ST>
__global__ void __launch_bounds__(W * 32, W > 4 ? 1 : 2) icl_attention_kernel(const __grid_constant__ Params p) {
  Kernel<W, MT, ST>::run(p);
}

// 8 warps x 2 m16 tiles = 256 query rows per CTA, 2-stage K/V ring, up to 4 key splits in the last wave.
constexpr int kWarps = 8, kTiles = 2, kStages = 2, kMaxSplits = 4;
using Cta = Kernel<kWarps, kTiles, kStages>;

using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                            const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                            CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

// [B, S, H, 64] -> boxes of `rows` x 64 for one (head, batch), 128B swizzle.
CUtensorMap tensor_map(const at::Tensor& t, int rows) {
  static Encode encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
    TORCH_CHECK(e == cudaSuccess && fn, "icl_attention: cuTensorMapEncodeTiled is unavailable");
    return reinterpret_cast<Encode>(fn);
  }();
  CUtensorMap map;
  cuuint64_t dims[4] = {D, (cuuint64_t)t.size(2), (cuuint64_t)t.size(1), (cuuint64_t)t.size(0)};
  cuuint64_t strides[3] = {(cuuint64_t)t.stride(2) * 2, (cuuint64_t)t.stride(1) * 2, (cuuint64_t)t.stride(0) * 2};
  cuuint32_t box[4] = {D, 1, (cuuint32_t)rows, 1}, unit[4] = {1, 1, 1, 1};
  CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), dims, strides, box, unit,
                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "icl_attention: cuTensorMapEncodeTiled failed with ", (int)r);
  return map;
}

void check(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dtype() == at::kBFloat16 && t.dim() == 4, "icl_attention: ", name,
              " must be a 4-d CUDA bf16 tensor");
  TORCH_CHECK(t.size(3) == D && t.stride(3) == 1, "icl_attention: ", name, " must have head dim 64 with unit stride");
  TORCH_CHECK(t.stride(0) % 8 == 0 && t.stride(1) % 8 == 0 && t.stride(2) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              "icl_attention: ", name, " rows must be 16-byte aligned");
}

// q [B, Sq, H, 64], k/v [B, Skv, H, 64], out [B, Sq, H, 64].
at::Tensor run(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor out, double scale) {
  check(q, "q");
  check(k, "k");
  check(v, "v");
  check(out, "out");
  TORCH_CHECK(k.sizes() == v.sizes() && q.size(0) == k.size(0) && q.size(2) == k.size(2) && out.sizes() == q.sizes(),
              "icl_attention: k and v must match, q must share their batch and heads, out must match q");
  const auto kernel = icl_attention_kernel<kWarps, kTiles, kStages>;
  Params p{};
  p.o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ob = out.stride(0), p.os = out.stride(1), p.oh = out.stride(2);
  p.sq = q.size(1), p.skv = k.size(1);
  p.heads = q.size(2);
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  if (p.sq == 0 || q.size(0) == 0 || p.heads == 0) return out;
  TORCH_CHECK(p.skv > 0, "icl_attention: k must have at least one row");
  p.nblk = (p.sq + Cta::BM - 1) / Cta::BM;
  p.tiles = (p.skv + BN - 1) / BN;
  const int64_t blocks = q.size(0) * p.heads * (int64_t)p.nblk;
  TORCH_CHECK(blocks * p.tiles < (1ll << 31), "icl_attention: blocks x key tiles must fit in int32");
  p.tq = tensor_map(q, Cta::BM);
  p.tk = tensor_map(k, BN);
  p.tv = tensor_map(v, BN);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cta::SMEM);
  // The last wave plus the remainder run as `splits` key ranges each, so
  // the tail of the launch is filled with short CTAs.
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, Cta::THREADS, Cta::SMEM);
  const int64_t wave = (int64_t)std::max(per_sm, 1) * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  p.splits = std::max(1, std::min(kMaxSplits, p.tiles / 96));
  p.whole = blocks;
  at::Tensor ws, counts;
  if (p.splits > 1) {
    p.whole = blocks % wave ? std::max<int64_t>(blocks - blocks % wave - wave, 0) : blocks;
    const int64_t parts = (blocks - p.whole) * p.splits;
    if (parts) {
      ws = at::empty({parts * Cta::THREADS * Cta::PART}, q.options().dtype(at::kFloat));
      p.ws = ws.data_ptr<float>();
      counts = at::zeros({blocks}, q.options().dtype(at::kInt));
      p.counts = counts.data_ptr<int>();
    }
  }
  const int64_t grid = p.whole + (blocks - p.whole) * p.splits;
  TORCH_CHECK(grid < (1ll << 31), "icl_attention: grid must fit in int32");
  kernel<<<static_cast<unsigned>(grid), Cta::THREADS, Cta::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace icl_attention

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &icl_attention::run); }
