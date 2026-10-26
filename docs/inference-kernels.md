# Inference kernels for KumoTabular

SM120 (Blackwell) CUDA kernels that speed up fresh-context `fit` + `predict` of the KumoTabular classifier. Kernel authors should read [`inference-kernel-interface.md`](inference-kernel-interface.md). The benchmark is described in [`benchmark/inference/README.md`](../benchmark/inference/README.md), and the latest measurements are in [`inference-kernel-results.md`](inference-kernel-results.md).

## How the kernels get into the model

SDM source is not changed. The model's `forward` is compiled with `torch.compile`, and one extra Inductor pass (`benchmark/inference/kernels/fx_pass.py`) runs on the post-grad graph, the ATen graph before Inductor fuses anything.

1. Each kernel family has a rule that matches the ATen pattern compiled SDM produces for one computation, for example an `addmm` followed by the decomposed erf-GELU.
2. The rule asks the registry whether a kernel admits the operands' shapes, dtypes, strides and device. Only compile-time metadata is used, so the choice is made before anything runs.
3. If a kernel admits them, the region is replaced by one custom op (`sdm_inference::<op>`) that calls that kernel. Otherwise the graph is left alone and Inductor compiles the native code.

Every kernel except ICL attention reproduces compiled native SDM bit for bit on its inputs: the same rounding points, the same GEMM reduction order as the cuBLAS kernel it replaces, and the same RMS arithmetic as SDM's Triton `rmsnorm_cast`. `--shadow` checks each kernel on the real model against the native value it replaces.

Whether the whole model then stays bitwise equal to compiled SDM depends on the op:

- **Bitwise end to end:** `linear_gelu`, `gated_log_scale`, `rope_rms`, `rope_rms_uneven`, `linear_kv_key_rms`, `linear_query_rms`, `column_projection_rms`, `icl_query_rms_scale`, `icl_key_rms`. They are on by default and the benchmark gates them bitwise.
- **ICL attention** (`icl_attention`): the BF16 kernel is close to compiled SDM but not bitwise; see [Default on or opt-in](#default-on-or-opt-in). The 8-bit kernel is lossy and opt-in. `SDM_ICL_ATTENTION=off` or `kernels.enable(model, exact=True)` keeps attention native.

**No row-table ops.** SDM's row embedding updates a table `[C, F+4, D]` in place. Eight ops that fused the residual adds, norms and down-projections reading or writing this table (`strided_rmsnorm`, `column_residual_rms`, `dual_rmsnorm`, `row_residual_rms`, `linear_residual_dual_rmsnorm`, `row_down_residual`, `row_projection_rms`, `linear_residual_kv_query_rms`) are not included. Each matched native bit for bit on native's inputs, but on SDM 890521bfa, where the row embedding is one compiled graph, the model was no longer bitwise end to end, and Large at 25k and 100k ran no faster than with the exact set. They moved the probabilities by a max |Δprob| of 0.144 at Large 100k, against 0.0075 for compiled vs eager. The cause is which values Inductor stores. Native code never stores the table after a column block (`(a + t) + c`): the norms read Inductor's private copies, and the row residual recomputes the table in FP32. A custom op that takes the table makes Inductor store it as BF16, so the residual adds the rounded table (in predict, the reverse).

The time saved comes from two sources:

- **Fewer round trips to memory.** A GEMM followed by a pointwise op (GELU, residual add, RMSNorm) normally writes a large intermediate tensor and reads it back. A fused epilogue never stores it.
- **Fewer temporaries.** Some native regions copy a strided view into a contiguous buffer before a norm. The fused kernel reads the strided view directly.

Notation:

- `C` is context rows and `F` is cell features after preprocessing (Covertype: `F = 53`). Each table row also has 4 readout tokens, so it has `F + 4` entries.
- `D` is the embedding width: 256 for Large and Medium, 128 for Small.
- `M` is the row count a GEMM or norm sees, for example `F * C` in the column stage.

## Families

Each family has one Python file in `benchmark/inference/kernels/fusions/` and its CUDA code in `benchmark/inference/kernels/csrc/`; the [op table](inference-kernel-interface.md#ops) lists them. The contribution figures were measured on SDM 842c408fe with torch 2.13.0, together with the row-table ops (see [Measured on SDM 842c408fe](#measured-on-sdm-842c408fe)); they have not been re-measured on 890521bfa.

### `linear_gelu`: linear + exact GELU

- **Replaces.** Every `Linear` followed by `GELU`: the embedding MLP up-projections (column, row and inducing blocks), the first layer of the row block's gating MLP, and the dataset (ICL) feed-forward. Native compiled code is a cuBLAS `addmm` producing a BF16 `[M, N]` tensor, then a Triton kernel that reads it, computes `0.5·x·(1 + erf(x/√2))` in FP32 and writes BF16.

- **Fuses.** GELU becomes the GEMM's epilogue, so the BF16 pre-activation is never stored or re-read.

- **Kernels.** `sm120_linear_gelu_k<K>_n<N>`. The embedding MLP and dataset FFN kernels are in `linear_gelu_sm120/linear_gelu.cu`; the first gating layer runs the first-layer kernel of `gating_sm120/gating.cu`.

  | Shape (K→N) | Use                | Models        |
  | ----------- | ------------------ | ------------- |
  | 256→512     | embedding MLPs     | Large, Medium |
  | 64→64       | first gating layer | Large, Medium |
  | 1024→2048   | dataset FFN        | Large only    |
  | 128→256     | embedding MLPs     | Small         |
  | 32→64       | first gating layer | Small         |

- **Exactness.** CUTLASS GEMMs with split-K 1 reproduce cuBLAS's reduction order, the bias is rounded once, and the GELU arithmetic is transcribed from Triton's libdevice. Below M = 1024 cuBLAS can choose a different reduction (it does for 256→512 at M = 64), so smaller query-side calls stay native.

- **Limits.** 1024 ≤ M ≤ the largest M checked bitwise: 8.39M for 256→512, 40M for 64→64, 500k for 1024→2048 (the dataset FFN sees M = C, at most 65,535). The Small kernels admit up to 33.0M (128→256) and 132M (32→64).

- **Contribution (Large 50k).** 2.1% (71 ms) on its own, the first row of the cumulative table below.

### `gated_log_scale`: second gating layer + query scaling

- **Replaces.** `GatedLogScale` in the row block's attention: a 64→head linear, then `q · log(key_len) · head_scale · (1 + tanh(z))`. Native code is an `addmm` plus one Inductor pointwise kernel.
- **Fuses.** The tanh gate and the query multiply become the GEMM epilogue (`sm120_gated_log_scale`).
- **Shapes.** Head 64 (Large, Medium) or head 32 (Small).
- **Exactness.** Bitwise equal to cuBLAS's WMMA kernel plus Inductor's pointwise kernel.
- **Limits.** 1024 ≤ M ≤ 80.9M (head 32) or 49.6M (head 64).
- **Contribution (Large 50k).** +0.7% (20 ms).

### `rope_rms`: RoPE + per-head RMSNorm for Q and K

- **Replaces.** The row block's `_RoPERMSNorm` on queries and keys. Compiled SDM applies rotary embedding as an Inductor pointwise kernel, then runs `rmsnorm_cast` per head: two passes for Q and two for K.
- **Fuses.** One launch rotates and normalises both Q and K; the rule pairs a layer's Q and K regions.
- **Shapes.** 4 heads of 64 (Large, Medium) or 32 (Small).
- **Exactness.** FP32 sin/cos with a single rounding, as compiled native.
- **Limits.** At most 1024 positions per row, and C·S ≤ 20.2M tokens (head 32) or 8.39M (head 64).
- **Contribution (Large 50k).** +1.3% (42 ms).

### `rope_rms_uneven`: RoPE + RMSNorm in the last row layer

- **Replaces.** RoPE and the per-head norms of the last row layer, whose queries are only the 4 readout tokens. Q has 4 positions and K has F + 4, so the `rope_rms` pairing does not apply.
- **Fuses.** Rotates and normalises Q and K of different lengths in one launch. Same module as `rope_rms`.
- **Shapes and limits.** Those of `rope_rms`, on the longer of Q and K.

### `linear_kv_key_rms`: KV projection + key RMSNorm

- **Replaces.** The column cross-attention's KV projection. Native code is an `addmm` into packed `[M, 2D]`, a split, a contiguous copy of K, and `rmsnorm_cast` on it.
- **Fuses.** The key norm runs in the GEMM epilogue, and raw K is never stored. V keeps its native packed layout.
- **Shapes.** D = 256 with head 64, or D = 128 with head 32.
- **Limits.** M ≤ 19.9M (D=128) or 12.0M (D=256). At D = 256, M must be at least 129, because cuBLAS sums smaller M in another order.
- **Contribution (Large 50k).** +1.2% (41 ms).

### `linear_query_rms`: query projection + query RMSNorm

- **Replaces.** The column cross-attention's query projection plus its per-head norm.
- **Fuses.** The norm runs in the epilogue; the raw query is never stored.
- **Limits.** M ≤ 33.0M (D=128) or 20.6M (D=256). A band of small M where cuBLAS reduces differently stays native.
- **Contribution (Large 50k).** +0.6% (20 ms).

### `column_projection_rms`: attention output projection + residual + RMSNorm

- **Replaces.** The column block's attention output projection, the residual add of `table[:, 4:]`, and the following norm.
- **Fuses.** All three run in one GEMM epilogue. The raw projection is still returned where later code reads it.
- **Limits.** M ≤ 19.8M (D=128) or 12.0M (D=256).

### `icl_query_rms_scale`, `icl_key_rms`: ICL query and key norms without the copies

- **Replaces.** Around each dataset (ICL) attention: the per-head query RMSNorm followed by a separate LogScale multiply `q · log(n) · head_scale`; the key RMSNorm, which `rmsnorm_cast` runs on a contiguous copy of the K view of the packed KV projection; and the copies of the K/V heads kept in the fitted cache.
- **Fuses.** The query norm and LogScale run in one pass. The key norm reads K in place from the packed projection and writes the cached K/V heads contiguous in the same launch. In the last fit layer, where only the cache reads K and V, the rule narrows the KV projection to the weight rows of the cached heads (2 of 16 on Large).
- **Shapes.** Head 64; ICL width 1024 (Large) or 512 (Medium, Small).
- **Exactness.** Native rounds the RMS to BF16, then computes the LogScale product in FP32 with one rounding; the kernel does the same. The narrowed projection sums in the same order as the full one for M from 513 to 65,535 (every M checked up to 8,192, then every 7th).
- **Limits.** The norms have no GEMM, so they do not depend on cuBLAS, and they index in 64 bits. The narrowed last-layer projection is used only for M in 513..65,535; outside that range the full projection runs and the kernel still does the norms and cache heads.
- **Contribution (Large 50k).** Profiler, fit: query side 12.5 to 6.3 ms, key side and cache slices 13.7 to 7.2 ms, last-layer projection 2.35 to 0.27 ms.

### `icl_attention`: ICL fit attention (not bitwise)

- **Replaces.** The dataset (ICL) attention in `fit`: context self-attention, R x R, head dim 64, 16 heads on Large and 8 on Medium and Small, 23 calls per fit. Compiled SDM runs PyTorch's flash attention, which is about 56% of fit + predict at Large 50k.
- **Does.** Exact softmax attention in a custom kernel (`sm120_icl_attention_d64`): 256 query rows per CTA with no register spills (flash spills), TMA loads into a 2-stage ring with per-stage barriers, online softmax that rescales only when a row maximum grew, and the launch's last wave split along keys and merged in a fixed order. The card runs at its power cap, so the gain comes from doing less non-MMA work per FLOP.
- **Exactness.** Not bitwise: it sums in a different order than flash. Outputs are deterministic from run to run. The benchmark qualifies it with the tolerance gate (below).
- **Shapes.** BF16, head dim 64, self-attention (same Q and K length) with 1,024 to 65,535 rows, no mask or dropout. Predict calls (16 query rows against the cache) and the row-block attention stay native. 65,535 is the largest context at which its error and determinism were checked; above it ICL attention stays native. Uncapped, Medium at 100k missed the tolerance gate narrowly (max |Δprob| 0.00952 against compiled-vs-eager 0.00948, 100% class agreement), so raising the cap needs the accuracy study at those sizes.
- **Switch.** `SDM_ICL_ATTENTION=sm120` (default), `fa2` (FlashAttention 2 at the same site, if installed), `int8` or `int8+ends` (below), or `off`; `kernels.enable(model, exact=True)` is the same as `off`. `sm120+uncapped` lifts the row cap, for diagnosis only.
- **Contribution.** Attention op alone, Large 50k: 5.7% (3,405 to 3,209 ms); Medium 4.4%, Small 5.3%. On top of every exact kernel, Large goes from 16.0% / 14.0% / 10.1% to 16.9% / 18.0% / 15.9% faster than compiled at 5k / 25k / 50k.
- **Evidence.** The study on 842c408fe covered 5 classification and 6 regression tables. Harnesses: `benchmark/attention/{accuracy,regression,determinism,compare}.py`. Results on 890521bfa are in [`inference-kernel-results.md`](inference-kernel-results.md#icl-attention-accuracy).

### Opt-in 8-bit ICL attention (`SDM_ICL_ATTENTION=int8`)

- **Does.** Quantizes K to INT8 per 64-key block after subtracting the per-head K mean (smoothing), V to FP8 E4M3 per channel and Q to INT8 per row; Q·Kᵀ runs on the INT8 `mma.sync` path, softmax in FP32, P·V in FP8 with FP32 accumulation. Source: `benchmark/inference/kernels/csrc/icl_attention_int8_sm120/icl_attention_int8.cu`, kernel `sm120_int8_icl`. The K mean is summed in a fixed order, so outputs are deterministic.
- **Settings.** `int8`, or `int8+ends`, which keeps the first and last ICL fit call of each graph native.
- **Gate.** Lossy by design (`exact=False`), never selected by default. The benchmark records the distance to compiled (and compiled vs eager when the eager arm runs) as gate `opt-in` and times the run.
- **Shapes.** BF16 or FP16, batch 1, head dim 64, at least 1,024 rows, 32-bit element offsets. Classification with more than 10 classes runs its ECOC codes as an ICL batch above 1 (8 for 26 classes), so attention stays native there; `benchmark.attention.compare` skips the 8-bit settings on those tables.
- **Measured on SDM 842c408fe** (torch 2.13): 1.6 to 1.7x flash standalone; Large 50k fit + predict 23.6% faster with the 8-bit attention alone and 33.8% with every exact kernel; class flips 0.16% at 1,024 query rows (compiled vs eager 0.92%); on TabArena, 1 of 74 per-task tests significant after correction (+0.014% log-loss).

## Default on or opt-in

The BF16 ICL attention kernel is on whenever the kernels are on. Whether it should stay on by default is for the SDM maintainers to decide. The rule used here for a default-on kernel that is not bitwise: the maximum probability difference against compiled SDM must stay within compiled SDM's own difference from eager SDM, with 100% class agreement, at 16 and at 1,024 query rows, on several datasets and seeds. The kernel meets that rule at 16 query rows. It does not meet it strictly at 1,024 query rows, or on the most extreme regression quantiles. The evidence below was measured on SDM 842c408fe; the 890521bfa study is in [`inference-kernel-results.md`](inference-kernel-results.md#icl-attention-accuracy).

| Evidence                                                         | Result                                                                                                                                                    |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bitwise equal to compiled                                        | No                                                                                                                                                        |
| 16 query rows, 4 classification datasets x 3 model/seed settings | 100% class agreement in 12 of 12                                                                                                                          |
| 1,024 query rows, same cases                                     | 4 of 12 flip 1 to 3 of 1,024 classes; compiled vs eager flips more rows in each (4 to 17); FlashAttention 2 flips the same rows in 3 of the 4             |
| Regression point prediction, 6 tables x 2 query counts x 3 seeds | inside the compiled-vs-eager gap in 36 of 36; RMSE shift at most 1.0% (eager vs compiled: 7.1%)                                                           |
| Regression per-quantile max                                      | over the gap in 3 of 36 runs, on one or two cells of the 0.001 or 0.002 quantile; the 99.9th percentile of \|Δ\| is 1.7 to 15x below the gap in every run |
| Run-to-run determinism                                           | bitwise identical                                                                                                                                         |
| Large, all kernels vs compiled, 5k / 25k / 50k                   | 16.9% / 18.0% / 15.9% with it, 16.0% / 14.0% / 10.1% without                                                                                              |

To make the kernels bitwise: `SDM_ICL_ATTENTION=off`, or `kernels.enable(model, exact=True)`.

## Measured on SDM 842c408fe

On SDM 842c408fe the compiled forward was split into several graphs around the row embedding and nothing was chunked under compile. Fresh-context `fit` + `predict`, clean compiled SDM vs the same plus all exact kernels without `icl_attention` (its numbers are above). Settings: one RTX PRO 4500 Blackwell, torch 2.13.0+cu130, BF16, Covertype, Q16, 12 mirrored blocks per run plus an unchanged repeat. Every run was bitwise equal to clean compiled on the probabilities and the fitted cache. These kernel sets include the row-table ops, which were bitwise end to end on that revision. The table shows the initial runs; the repeats differ by at most 0.11 percentage points.

| Model  | 5k    | 25k   | 50k   |
| ------ | ----- | ----- | ----- |
| Large  | 15.2% | 13.1% | 9.5%  |
| Medium | 16.9% | 17.4% | 13.9% |
| Small  | 9.0%  | 12.9% | 10.6% |

Large 50k, adding one family at a time (short 4-block runs, attribution only):

| Kernels                                         | Faster than compiled |
| ----------------------------------------------- | -------------------- |
| `linear_gelu`                                   | 2.10%                |
| + `strided_rmsnorm`                             | 3.20%                |
| + `gated_log_scale`                             | 3.91%                |
| + `rope_rms`                                    | 5.17%                |
| + `linear_kv_key_rms`                           | 6.38%                |
| + `column_residual_rms`, `dual_rmsnorm`         | 7.37%                |
| + `linear_query_rms`                            | 7.98%                |
| + in-place column-down and row residual         | 8.40%                |
| + `row_down_residual`                           | 8.90%                |
| + `column_projection_rms`, `row_projection_rms` | 9.67%                |

Adding the attention-adjacent ops (`icl_query_rms_scale`, `icl_key_rms`, `rope_rms_uneven` and the row-table op `linear_residual_kv_query_rms`) makes Large 50k about 10.2% faster than compiled, against about 9.5% with the set above in the same process. This is a same-process A/B of compiled, the set above and every op, 12 interleaved blocks plus an unchanged repeat, all bitwise equal. The attention-adjacent ops add 0.3 to 1.1% across Large, Medium and Small at 5k to 50k.

## Coverage and limits

Compiled SDM runs any context: SDPA splits flash calls above 65,535 batch entries (SDM #1181), and attention and cell embedding chunk under `torch.compile` with a memory budget per chunk (SDM #1055, `SDM_CHUNK_MEMORY_FRACTION`, default 5% of the GPU). No flash call in the graph has more than 65,535 batch entries.

Chunking changes what the rules see:

- **Row blocks** run on chunks of a few thousand contexts (about 7,200 for Large at 53 features on a 32 GB GPU) from about 7,000 contexts up. The row-local ops (`rope_rms`, `rope_rms_uneven`, gating, the GEMM epilogues) match every chunk, and no upper bound binds at 100k contexts.
- **Column blocks** run on chunks of a few features once the context is above about 7,500 rows (Large and Medium; about 15,000 for Small). A chunk reads a feature range of the table, which `column_projection_rms` does not take, so that region stays native at large contexts. `linear_kv_key_rms` and `linear_query_rms` work on the chunk's own GEMM and still match.
- **ICL block** is not chunked: its ops see M = C. `icl_key_rms` narrows the last fit layer's KV projection only for M in 513..65,535, where it was checked bitwise; above that it runs the full projection and still does the norms and cache heads. The BF16 `icl_attention` kernel admits at most 65,535 rows (checked); the 8-bit kernel has no row cap.

| Family                               | Models                    | Rows admitted (M)                                                                     | Outside the limits                                           |
| ------------------------------------ | ------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `linear_gelu`                        | all; 1024→2048 Large only | 1024 to 8.39M (256→512), 40M (64→64), 500k (1024→2048), 33.0M (128→256), 132M (32→64) | native                                                       |
| `gated_log_scale`                    | all                       | 1024 to 80.9M (head 32), 49.6M (head 64)                                              | native                                                       |
| `rope_rms`                           | all                       | ≤ 1024 positions, ≤ 20.2M tokens (head 32), 8.39M (head 64)                           | native                                                       |
| `linear_kv_key_rms`                  | all                       | ≤ 19.9M (D=128), 12.0M (D=256)                                                        | native                                                       |
| `linear_query_rms`                   | all                       | ≤ 33.0M (D=128), 20.6M (D=256), minus a small-M band                                  | native                                                       |
| `column_projection_rms`              | all                       | 19.8M (D=128), 12.0M (D=256)                                                          | native                                                       |
| `icl_query_rms_scale`, `icl_key_rms` | all                       | no row bound; narrowed last-layer KV projection for M in 513..65,535                  | full projection (outside the M range), native (other shapes) |
| `rope_rms_uneven`                    | all                       | as `rope_rms`, on the longer of Q and K                                               | native                                                       |
| `icl_attention` BF16 (not bitwise)   | all                       | self-attention of 1,024 to 65,535 rows, head dim 64                                   | native                                                       |
| `icl_attention` int8 (opt-in, lossy) | all                       | self-attention of at least 1,024 rows, head dim 64, batch 1                           | native                                                       |
