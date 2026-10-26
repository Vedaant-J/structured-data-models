# Inference kernel results

Speed and accuracy of the kernels described in [`inference-kernels.md`](inference-kernels.md), measured on SDM `main` at 890521bfa with torch 2.14.1+cu130. Three kernel sets are compared:

- the **exact fused kernels**, bitwise equal to compiled SDM end to end;
- the **BF16 SM120 ICL attention** kernel (default `--attention sm120`; close to compiled but not bitwise);
- the **opt-in 8-bit ICL attention** (`--attention int8`; lossy).

Setup for every speed number: one RTX PRO 4500 Blackwell (SM 12.0, 165 W power cap), BF16 autocast, fresh-context `fit` + `predict` with 16 query rows on Covertype, `torch.compile(dynamic=False)`. The benchmark is described in [`benchmark/inference/README.md`](../benchmark/inference/README.md).

| Arm           | What runs                                                          | Flag                                                                |
| ------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| a             | compiled SDM                                                       | `compiled`                                                          |
| b             | a + exact kernels, attention native                                | `optimized:off`                                                     |
| c             | b + BF16 ICL attention                                             | `optimized:sm120` (capped at 65,535 rows, so c = b at 75k and 100k) |
| d             | b + 8-bit ICL attention                                            | `optimized:int8`                                                    |
| int8+ends     | b + 8-bit attention, first and last ICL call of each graph in BF16 | `optimized:int8+ends` (Large 50k and 100k)                          |
| bf16 uncapped | c without the row cap (diagnostic)                                 | `optimized:sm120+uncapped` (Large 100k)                             |
| eager         | eager SDM                                                          | `eager` (5k and 50k)                                                |

All arms of a cell run interleaved in one process: 12 timing blocks, or 15 for cells with 5 arms, in a balanced (Williams square) order, after 3 untimed warmup calls per arm. Large 50k and 100k were run a second time unchanged.

Large, against compiled: exact kernels -11.1% / -8.4% / -6.0% / -4.6% / -3.7% at 5k / 25k / 50k / 75k / 100k; with BF16 attention -11.9% / -12.4% / -11.3% up to 50k; with 8-bit attention -14.4% / -23.6% / -28.4% / -31.2% / -33.1%. The exact kernels also lower peak allocated memory on Large from 25k up and on Medium from 75k up (Large 100k: 15.5 to 9.6 GiB).

## Speed

Each cell: median ms, change vs compiled (negative is faster), peak allocated GiB. Power and SM clock are medians over the timed window.

### Large

| Context     | a compiled   | b exact             | c +bf16 attn        | d +int8 attn        | int8+ends           | bf16 uncapped       | eager              | Power W | SM MHz |
| ----------- | ------------ | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------ | ------- | ------ |
| 5k          | 207, 2.1     | 184 (-11.1%), 2.2   | 183 (-11.9%), 2.2   | 177 (-14.4%), 2.2   |                     |                     | 230 (+10.8%), 2.2  | 70      | 2,400  |
| 25k         | 1,264, 4.5   | 1,157 (-8.4%), 4.1  | 1,108 (-12.4%), 4.1 | 965 (-23.6%), 4.1   |                     |                     |                    | 113     | 2,126  |
| 50k         | 3,521, 8.1   | 3,310 (-6.0%), 5.5  | 3,122 (-11.3%), 5.5 | 2,520 (-28.4%), 5.5 | 2,588 (-26.5%), 5.5 |                     | 3,572 (+1.5%), 4.1 | 165     | 1,792  |
| 50k repeat  | 3,536, 8.1   | 3,325 (-6.0%), 5.5  | 3,134 (-11.4%), 5.5 | 2,530 (-28.5%), 5.5 | 2,597 (-26.5%), 5.5 |                     | 3,587 (+1.4%), 4.1 | 165     | 1,785  |
| 75k         | 6,824, 11.7  | 6,508 (-4.6%), 7.5  | = b (cap)           | 4,695 (-31.2%), 7.5 |                     |                     |                    | 165     | 1,770  |
| 100k        | 11,131, 15.5 | 10,721 (-3.7%), 9.6 | = b (cap)           | 7,445 (-33.1%), 9.6 | 7,725 (-30.6%), 9.6 | 9,935 (-10.7%), 9.6 |                    | 165     | 1,762  |
| 100k repeat | 11,102, 15.5 | 10,688 (-3.7%), 9.6 | = b (cap)           | 7,425 (-33.1%), 9.6 | 7,703 (-30.6%), 9.6 | 9,910 (-10.7%), 9.6 |                    | 165     | 1,770  |

### Medium

| Context | a compiled  | b exact            | c +bf16 attn        | d +int8 attn        | eager              | Power W | SM MHz |
| ------- | ----------- | ------------------ | ------------------- | ------------------- | ------------------ | ------- | ------ |
| 5k      | 184, 1.5    | 162 (-12.1%), 1.7  | 161 (-12.7%), 1.7   | 158 (-14.1%), 1.7   | 201 (+9.4%), 1.7   | 60      | 2,407  |
| 25k     | 955, 3.5    | 856 (-10.4%), 3.5  | 830 (-13.1%), 3.5   | 756 (-20.8%), 3.5   |                    | 114     | 2,070  |
| 50k     | 2,443, 5.0  | 2,243 (-8.2%), 5.0 | 2,145 (-12.2%), 5.0 | 1,850 (-24.3%), 5.0 | 2,398 (-1.9%), 3.5 | 147     | 1,815  |
| 75k     | 4,477, 7.8  | 4,175 (-6.7%), 6.9 | = b (cap)           | 3,275 (-26.8%), 6.9 |                    | 165     | 1,792  |
| 100k    | 7,067, 12.1 | 6,670 (-5.6%), 9.2 | = b (cap)           | 5,046 (-28.6%), 9.2 |                    | 165     | 1,770  |

### Small

| Context | a compiled | b exact            | c +bf16 attn      | d +int8 attn        | eager             | Power W | SM MHz |
| ------- | ---------- | ------------------ | ----------------- | ------------------- | ----------------- | ------- | ------ |
| 5k      | 89, 0.8    | 82 (-8.0%), 0.9    | 82 (-8.5%), 0.9   | 80 (-9.7%), 0.9     | 100 (+11.8%), 1.0 | 59      | 2,377  |
| 25k     | 370, 1.9   | 332 (-10.1%), 1.9  | 320 (-13.4%), 1.9 | 286 (-22.8%), 1.9   |                   | 103     | 2,257  |
| 50k     | 933, 3.2   | 861 (-7.7%), 3.2   | 814 (-12.8%), 3.2 | 675 (-27.7%), 3.2   | 963 (+3.2%), 3.0  | 126     | 2,014  |
| 75k     | 1,739, 3.6 | 1,630 (-6.3%), 3.6 | = b (cap)         | 1,204 (-30.8%), 3.6 |                   | 134     | 1,856  |
| 100k    | 2,767, 5.0 | 2,620 (-5.3%), 4.9 | = b (cap)         | 1,863 (-32.7%), 4.9 |                   | 162     | 1,815  |

No cell ran out of memory. Every b run was bitwise equal to compiled (probabilities and every fitted-cache tensor), and every c run passed the tolerance gate with 100% class agreement. The profiler found every kernel of each arm in each cell.

The card is at its 165 W cap for Large from 50k up, Medium from 75k up and Small at 100k (about 1.8 GHz). Smaller cells run below the cap at higher clocks (up to 2.4 GHz), so their percentages are not power-bound. At large contexts the fixed-cost savings of the exact kernels shrink relative to ICL attention, which grows with the square of the context; the 8-bit attention gains more as the context grows.

### Speed vs accuracy

Max |Δprob| against compiled on the 16 timed query rows of each cell, with compiled vs eager on the same rows as the reference gap (eager is run untimed for the comparison where it is not timed). Repeats give identical distances.

| Model  | Context | compiled vs eager | c BF16 | d int8 | int8+ends | BF16 uncapped |
| ------ | ------- | ----------------- | ------ | ------ | --------- | ------------- |
| Large  | 5k      | 0.0333            | 0.0058 | 0.0135 |           |               |
| Large  | 25k     | 0.0224            | 0.0032 | 0.0119 |           |               |
| Large  | 50k     | 0.1063            | 0.0110 | 0.0338 | 0.0057    |               |
| Large  | 75k     | 0.0117            |        | 0.0059 |           |               |
| Large  | 100k    | 0.0075            |        | 0.0430 | 0.0148    | 0.0132        |
| Medium | 5k      | 0.0198            | 0.0090 | 0.0125 |           |               |
| Medium | 25k     | 0.0368            | 0.0056 | 0.0088 |           |               |
| Medium | 50k     | 0.0445            | 0.0017 | 0.0322 |           |               |
| Medium | 75k     | 0.0047            |        | 0.0044 |           |               |
| Medium | 100k    | 0.0095            |        | 0.0039 |           |               |
| Small  | 5k      | 0.0430            | 0.0074 | 0.0124 |           |               |
| Small  | 25k     | 0.0732            | 0.0019 | 0.0071 |           |               |
| Small  | 50k     | 0.0277            | 0.0092 | 0.0126 |           |               |
| Small  | 75k     | 0.0134            |        | 0.0031 |           |               |
| Small  | 100k    | 0.0929            |        | 0.0497 |           |               |

Class agreement was 100% for every arm in every cell. The 8-bit attention stays inside the gap except at Large 100k, where int8 is 5.7x and int8+ends 2x over a small gap; the 16-row gap varies a lot between cells, so the accuracy study below is the better guide.

## Effect of SDM #1055 on the compiled baseline

SDM #1055 makes attention and cell embedding chunk under `torch.compile` with a memory budget per chunk. Compiled Large 50k, compiled arm only, 2 processes x 6 blocks each:

| Build                                                           | Median ms | Peak allocated / reserved GiB |
| --------------------------------------------------------------- | --------- | ----------------------------- |
| (i) 890521bfa, default                                          | 3,468     | 8.13 / 11.26                  |
| (ii) 890521bfa, chunking off (`SDM_CHUNK_MEMORY_FRACTION=1000`) | 3,358     | 13.00 / 20.97                 |
| (iii) 842c408fe on torch 2.14.1                                 | 3,315     | 10.47 / 17.37                 |
| (iv) 890521bfa with only #1055 reverted                         | 3,353     | 13.00 / 20.97                 |

(ii) was checked to run as one chunk: its graph has 41 SDPA nodes against 149 with chunking. 842c408fe measured 3,313 ms on torch 2.13, so the torch version accounts for almost nothing. (iv) matches (ii), so #1055 alone costs about 110 ms (+3.2%) and saves 4.9 GiB allocated (-37%) and 9.7 GiB reserved. A fresh repeat of (i) after the sweep gave 3,460 ms. The remaining ~40 ms (1.1%) between (iv) and (iii) is other SDM changes. The first compiled call takes 165 s against 55 s on 842c408fe, with an FX graph of about 18,000 nodes against 5,000. Compiled Large 100k runs (11.1 s, 15.5 GiB allocated); on 842c408fe it fails, because flash attention rejects more than 65,535 batch entries.

## Correctness

### Exact kernels

- Kernel tests: 240 passed, 2 skipped (FlashAttention 2 not installed); 274 passed, 2 skipped with the row-table ops included.

- Bitwise end to end, default exact set (`--attention off`), every fitted-cache tensor and the probabilities (64 tensors; 36 for Small), Small, Medium and Large at 5k, 50k and 100k: all equal. Nine ops match at 5k and eight from 50k up (`column_projection_rms` drops out because columns are feature-chunked there).

  | Model  | Sites at 5k | 50k | 100k |
  | ------ | ----------- | --- | ---- |
  | Small  | 93          | 205 | 301  |
  | Medium | 153         | 513 | 921  |
  | Large  | 176         | 536 | 944  |

- `--shadow` (native stays on the data path, each kernel output compared bit for bit with the value it replaces): every op exact.

- SDM tests: `test/nn/test_attention.py` 16 passed; `FULL_TEST=1 test/nn/test_compile.py` 31 passed. Nothing under `sdm/` is changed.

### ICL attention accuracy

Large, 50,000 context rows (47,818 for adult, the whole table), 16 and 1,024 query rows, seeds 0 to 2. Each setting is compiled SDM plus only the attention op. Per cell: worst max |Δprob| over seeds vs compiled, class flips summed over the 3 seeds (48 or 3,072 rows), change in log-loss (mean over seeds).

| Dataset   | Query rows | compiled vs eager   | c BF16             | d int8             | int8+ends          |
| --------- | ---------- | ------------------- | ------------------ | ------------------ | ------------------ |
| adult     | 16         | 0.0186, 0, +0.0017  | 0.0098, 0, -0.0001 | 0.0122, 0, -0.0002 | 0.0054, 0, -0.0001 |
| adult     | 1,024      | 0.1372, 19, -0.0002 | 0.0045, 0, +0.0000 | 0.0094, 0, -0.0000 | 0.0103, 0, -0.0000 |
| covertype | 16         | 0.1125, 1, -0.0090  | 0.0137, 1, +0.0007 | 0.0338, 1, +0.0006 | 0.0234, 1, -0.0022 |
| covertype | 1,024      | 0.1761, 19, -0.0001 | 0.0155, 1, -0.0001 | 0.0915, 9, +0.0002 | 0.0407, 2, +0.0002 |
| higgs     | 16         | 0.0203, 1, +0.0005  | 0.0029, 0, -0.0001 | 0.0071, 0, +0.0004 | 0.0050, 0, -0.0006 |
| higgs     | 1,024      | 0.0814, 37, +0.0008 | 0.0063, 2, +0.0000 | 0.0100, 7, +0.0002 | 0.0092, 4, +0.0001 |
| volkert   | 16         | 0.0885, 0, +0.0108  | 0.0091, 0, +0.0013 | 0.0203, 0, +0.0062 | 0.0098, 0, +0.0052 |
| volkert   | 1,024      | 0.4468, 51, +0.0009 | 0.0199, 3, -0.0000 | 0.0566, 7, +0.0001 | 0.0500, 6, +0.0003 |

- **BF16 (c)** is inside the compiled-vs-eager gap in every run. At 16 query rows it flips one row: a covertype near-tie that eager flips too. At 1,024 rows it flips 0 to 3 of 3,072 rows (eager: 19 to 51). Log-loss moves by at most ±0.0013.
- **int8 (d)** is over the per-seed gap in 1 of 3 covertype 16-row seeds (0.0275 against 0.0098). At 1,024 rows it flips 0 to 0.29% of rows (eager 0.6 to 1.7%). Log-loss moves by -0.0002 to +0.0062.
- **int8+ends** is over the gap in 1 of 3 covertype 16-row seeds, flips fewer rows than int8, and moves log-loss by -0.0022 to +0.0052.

Two more classification tables, same layout. bank_marketing uses 44,187 context rows and letter 18,976, the whole table in each case. On letter the 8-bit settings stay native (see below).

| Dataset        | Query rows | compiled vs eager  | c BF16             | d int8             | int8+ends          |
| -------------- | ---------- | ------------------ | ------------------ | ------------------ | ------------------ |
| bank_marketing | 16         | 0.0248, 0, -0.0023 | 0.0037, 0, +0.0000 | 0.0062, 0, -0.0006 | 0.0035, 0, -0.0001 |
| bank_marketing | 1,024      | 0.0397, 6, -0.0001 | 0.0064, 0, +0.0000 | 0.0063, 2, -0.0000 | 0.0060, 0, -0.0000 |
| letter         | 16         | 0.0154, 0, +0.0010 | 0.0014, 0, -0.0001 | native             | native             |
| letter         | 1,024      | 0.0675, 0, +0.0001 | 0.0104, 0, +0.0000 | native             | native             |

Small and Medium, 50,000 context rows, seed 0 only, so the flips are out of 16 or 1,024 rows:

| Model  | Dataset   | Query rows | compiled vs eager   | c BF16             | d int8             | int8+ends          |
| ------ | --------- | ---------- | ------------------- | ------------------ | ------------------ | ------------------ |
| Medium | covertype | 16         | 0.0445, 0, -0.0009  | 0.0017, 0, -0.0001 | 0.0322, 0, -0.0017 | 0.0139, 0, -0.0022 |
| Medium | covertype | 1,024      | 0.1317, 3, -0.0014  | 0.0216, 2, +0.0002 | 0.0534, 1, +0.0005 | 0.0337, 3, +0.0004 |
| Medium | volkert   | 16         | 0.0644, 0, -0.0694  | 0.0079, 0, +0.0074 | 0.0137, 0, +0.0048 | 0.0115, 0, -0.0007 |
| Medium | volkert   | 1,024      | 0.1054, 13, -0.0017 | 0.0328, 1, +0.0003 | 0.0505, 6, +0.0005 | 0.0290, 4, +0.0009 |
| Small  | covertype | 16         | 0.0277, 0, -0.0004  | 0.0092, 0, -0.0017 | 0.0126, 0, -0.0004 | 0.0123, 0, -0.0004 |
| Small  | covertype | 1,024      | 0.2246, 11, -0.0002 | 0.0234, 2, +0.0004 | 0.0546, 4, +0.0002 | 0.0267, 1, +0.0001 |
| Small  | volkert   | 16         | 0.0521, 0, +0.0554  | 0.0061, 0, +0.0019 | 0.0256, 0, -0.0061 | 0.0114, 0, -0.0007 |
| Small  | volkert   | 1,024      | 0.1952, 23, +0.0025 | 0.0225, 1, +0.0000 | 0.0396, 5, +0.0003 | 0.0292, 2, -0.0004 |

- Every setting is inside the gap in every classification run, except the two covertype 16-row seeds noted above.
- Flips at 1,024 rows: BF16 flips 0 to 3 per cell, always fewer than eager. int8 flips 0 to 9 and int8+ends 0 to 6; the one cell where int8+ends equals eager is Medium covertype (3 against 3). At 16 rows no setting flips any row except the covertype near-tie.
- Log-loss changes by at most ±0.001 at 1,024 rows for every setting. At 16 rows the change is noisier, up to ±0.0074 (BF16 on Medium volkert, one seed, where eager moves -0.069).

Regression, Large, 6 TabArena tables (context: the table minus 1,024, at most 50,000), seeds 0 to 2. The model predicts 999 quantiles. Distances are |Δ| / the context target's standard deviation, against compiled. Per cell: worst over seeds of the max over all quantiles, worst point-prediction (quantile mean) |Δ|, and the change in RMSE (mean over seeds).

| Table                  | Context | Query rows | compiled vs eager     | c BF16                | d int8                | int8+ends             |
| ---------------------- | ------- | ---------- | --------------------- | --------------------- | --------------------- | --------------------- |
| diamonds               | 50,000  | 16         | 0.125, 0.0129, -1.13% | 0.063, 0.0032, +0.05% | 0.047, 0.0040, +0.39% | 0.063, 0.0043, +0.86% |
| diamonds               | 50,000  | 1,024      | 0.445, 0.0582, -0.55% | 0.305, 0.0171, -0.07% | 0.535, 0.0189, +0.04% | 0.367, 0.0181, +0.03% |
| food_delivery_time     | 44,427  | 16         | 0.814, 0.1828, +0.85% | 0.125, 0.0085, +0.02% | 0.178, 0.0105, -0.01% | 0.172, 0.0051, -0.01% |
| food_delivery_time     | 44,427  | 1,024      | 1.352, 0.5197, +0.06% | 0.266, 0.0291, +0.00% | 0.453, 0.0552, +0.00% | 0.328, 0.0466, -0.00% |
| houses                 | 19,616  | 16         | 0.375, 0.0837, -0.62% | 0.141, 0.0116, -0.40% | 0.219, 0.0300, -0.07% | 0.281, 0.0187, -0.35% |
| houses                 | 19,616  | 1,024      | 2.555, 0.2360, +0.25% | 0.637, 0.0270, -0.03% | 1.008, 0.0572, -0.01% | 0.773, 0.0491, -0.05% |
| physiochemical_protein | 44,706  | 16         | 1.248, 0.1921, +2.18% | 0.319, 0.0128, -0.13% | 0.383, 0.0442, +0.73% | 0.474, 0.0617, +0.37% |
| physiochemical_protein | 44,706  | 1,024      | 3.594, 0.4222, +0.42% | 1.098, 0.0465, -0.02% | 1.973, 0.1214, -0.05% | 1.701, 0.0979, -0.05% |
| qsar_tid_11            | 4,718   | 16         | 0.324, 0.0667, +0.24% | 0.141, 0.0076, -0.08% | 0.281, 0.0352, +0.03% | 0.195, 0.0164, +0.68% |
| qsar_tid_11            | 4,718   | 1,024      | 1.250, 0.1792, -0.20% | 0.531, 0.0243, +0.01% | 1.969, 0.0845, -0.01% | 0.965, 0.0408, -0.00% |
| superconductivity      | 20,239  | 16         | 0.281, 0.0502, +0.91% | 0.148, 0.0079, -0.16% | 0.234, 0.0230, +0.50% | 0.367, 0.0143, +0.12% |
| superconductivity      | 20,239  | 1,024      | 1.445, 0.2830, +0.67% | 0.703, 0.0563, -0.08% | 1.313, 0.1271, -0.23% | 0.875, 0.1293, -0.03% |

- **BF16 (c):** the max over all quantiles is inside the same seed's eager gap in 36 of 36 runs, and so is the point prediction. Within each band of quantiles (5% tails, 45 to 55% median) it passes eager's band max in 2 of 108 cases, both in the high tail (diamonds 16 rows: 0.063 against 0.031; qsar_tid_11 1,024 rows: 0.453 against 0.441). RMSE moves at most 1.1% in any seed (eager: up to 7.0%).
- **int8 (d):** the point prediction is inside the gap in 36 of 36 runs; the max over all quantiles in 31 of 36. The misses are on diamonds, qsar_tid_11 and superconductivity, in the tail quantiles (12 of 108 band cases). RMSE moves at most 2.4% in a seed.
- **int8+ends:** point prediction 36 of 36; all quantiles 33 of 36; 10 of 108 band cases over; RMSE at most 3.4% in a seed (diamonds, 16 rows).
- **letter:** with 26 classes ECOC runs the ICL attention as a batch of 8. The 8-bit kernel takes only batch 1 and refuses it ("needs batch 1 (the ICL fit site); got 8"), so attention stays native and d and int8+ends equal compiled. BF16 takes batch 8 and is inside the gap.

### Determinism

Large 50k, 10 runs each: compiled, BF16, int8 and int8+ends are each bitwise identical from run to run.

### Why the BF16 cap stays at 65,535 rows

Uncapped BF16 at Large 100k is 10.7% faster than compiled, but its max |Δprob| is 0.0132 against a gap of 0.0075. Medium 100k uncapped missed the gate narrowly (0.00952 against 0.00948). The kernel's error and determinism were checked up to 65,535 rows; raising the cap needs the accuracy study at those sizes.

## Coverage at these sizes

- `column_projection_rms` stays native from 50k up, because a feature chunk reads a range of the table it does not take (passing the range would recover about 0.4%). `linear_query_rms` stays native at Small 100k, whose M falls in the band where cuBLAS reduces differently. `icl_key_rms` narrows the last layer's KV projection only up to 65,535 rows.
- The eight row-table ops (see [`inference-kernels.md`](inference-kernels.md#how-the-kernels-get-into-the-model)) are excluded. Each matched native bit for bit on native's inputs, but with them on top of b, Large 25k ran 1,158 ms (b: 1,157) and Large 100k the same as b, while max |Δprob| at Large 100k was 0.144 against a gap of 0.0075.

## Reproduce

From the repository root on an SM 12.0 GPU, with a CUDA 13.0 toolkit for the kernel build:

```bash
# environment as in .github/workflows/test-gpu.yml (resolves torch 2.14.1+cu130)
uv pip install --torch-backend=cu130 --extra-index-url=https://pypi.nvidia.com \
    --find-links=https://data.pyg.org/whl/torch-2.14.0+cu130.html \
    --index-strategy=unsafe-best-match --group=test . scikit-learn pandas ninja

# speed sweep (resumable; --dry-run lists cells, arms and blocks)
python -m benchmark.inference.sweep run OUT --dry-run
python -m benchmark.inference.sweep run OUT
python -m benchmark.inference.sweep table OUT        # OUT/tables.md, OUT/table.csv

# one cell
python -m benchmark.inference --model-size large --context-rows 50000 \
    --arms compiled optimized:off optimized:sm120 optimized:int8 eager --blocks 15 --output runs/large-50k

# bitwise and shadow checks of the exact set
python -m benchmark.inference --model-size large --context-rows 50000 --attention off \
    --arms compiled optimized --qualify-only --output runs/bit-large-50k
python -m benchmark.inference --model-size large --context-rows 50000 --attention off \
    --arms compiled optimized --shadow --output runs/shadow-large-50k

# attention accuracy and determinism
python -m benchmark.attention.compare_data --out DATA
python -m benchmark.attention.compare run --data-dir DATA --out RES --settings sm120 int8 int8+ends \
    --contexts 50000 --queries 16 1024 --seeds 0 1 2
python -m benchmark.attention.compare tabulate --results RES --output RES/compare.md
python -m benchmark.attention.determinism --out det-int8.json --size large --rows 50000 --attention int8

# #1055 attribution: chunking off
SDM_CHUNK_MEMORY_FRACTION=1000 python -m benchmark.inference --model-size large --context-rows 50000 \
    --arms compiled --output runs/nochunk
```
