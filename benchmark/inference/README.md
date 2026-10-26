# KumoTabular inference latency

Measures the latency of `fit(context, labels)` + `predict(query_rows)` with a fresh context on a KumoTabular classifier, on one CUDA GPU.

## Prerequisites

- An NVIDIA GPU with a CUDA build of PyTorch and Triton, and SDM installed from this checkout, plus `scikit-learn`, `pandas` and `ninja`:

  ```bash
  python -m venv .venv && . .venv/bin/activate
  pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
  pip install -e . scikit-learn pandas ninja
  ```

  Results were measured with torch 2.14.1+cu130. Compiled code, and so the timings and the bitwise status of a run, depend on the torch version.

- For the `optimized` arm: an SM 12.0 (Blackwell) GPU, `ninja`, a CUDA toolkit whose `nvcc` matches torch's CUDA version (set `CUDA_HOME`, e.g. `/usr/local/cuda-13.0`, if it is not the first `nvcc` on `PATH`), and `git` to fetch CUTLASS at a pinned commit (or set `CUTLASS_ROOT` to a checkout). The first optimized run compiles the kernels in `benchmark/inference/kernels/csrc/`, which takes several minutes; later runs reuse them.

- Run from the repository root. scikit-learn downloads Covertype on first use. Without `--checkpoint`, the pretrained weights are downloaded from Hugging Face.

## Run

```bash
python -m benchmark.inference --model-size large --context-rows 50000 --output runs/large-50k
```

| Flag                 | Default                                                                | Meaning                                                                                                                                                                                                             |
| -------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model-size`       | required                                                               | `small`, `medium` or `large`                                                                                                                                                                                        |
| `--context-rows`     | required                                                               | context rows per call                                                                                                                                                                                               |
| `--output`           | required                                                               | new directory; existing directories are refused                                                                                                                                                                     |
| `--checkpoint`       | pretrained download                                                    | classifier state dict (`<size>/classifier.pt`)                                                                                                                                                                      |
| `--kernels`          | `$SDM_KERNEL_CACHE` or `~/.cache/sdm-inference-kernels/<torch>-<cuda>` | kernel module directory for `optimized`; missing or stale modules are built into it, also when a directory is given                                                                                                 |
| `--ops`              | all                                                                    | limit `optimized` to these named ops, for cumulative measurements; the names are listed in [`docs/inference-kernel-interface.md`](../../docs/inference-kernel-interface.md#ops)                                     |
| `--attention`        | `$SDM_ICL_ATTENTION` or `sm120`                                        | ICL attention kernel in `optimized`: `sm120` (BF16), `fa2` (FlashAttention 2, if installed), `off` (native), or the lossy opt-in `int8` or `int8+ends`; `sm120+uncapped` lifts the BF16 row cap, for diagnosis only |
| `--shadow`           | off                                                                    | run each selected kernel beside native on native's inputs and report per op whether its outputs are bitwise equal; no timing (status `shadow_bitwise` or `shadow_mismatch`)                                         |
| `--data`, `--target` | Covertype                                                              | a `.parquet` or `.csv` table and its categorical target column                                                                                                                                                      |
| `--query-rows`       | 16                                                                     | query rows per call                                                                                                                                                                                                 |
| `--device`           | `cuda:0`                                                               | CUDA device                                                                                                                                                                                                         |
| `--arms`             | all three                                                              | any of `eager compiled optimized`, plus `optimized:<attention>` arms (see Arms); optimized arms need `compiled`                                                                                                     |
| `--blocks`           | 12                                                                     | timing blocks; a multiple of the arm count (every order is used when the blocks are a multiple of their count)                                                                                                      |
| `--warmup`           | 3                                                                      | untimed calls per arm; the first includes compilation                                                                                                                                                               |
| `--precision`        | `ieee`                                                                 | `ieee` disables TF32 and reduced-precision BF16 reductions; `default` leaves PyTorch defaults                                                                                                                       |
| `--seed`             | 0                                                                      | seed of the row permutation and of the per-call CUDA generator                                                                                                                                                      |
| `--allow-mismatch`   | off                                                                    | time `optimized` even if it fails its gate against `compiled`                                                                                                                                                       |
| `--qualify-only`     | off                                                                    | warm up and qualify the arms, then stop before timing (status `qualified`)                                                                                                                                          |

Rows come from a seeded permutation: the context is its first rows and the query the rows right after.

The exit code is 0 when the run completed, 2 when it completed without `optimized` (skipped on a non-SM120 GPU, or when no kernel admits any region), and 1 otherwise (qualification failed, completed with an allowed mismatch, or a kernel build error).

## Arms

- `eager`: SDM as is, no compilation.
- `compiled`: SDM with the classifier's `forward` replaced by `torch.compile(forward, fullgraph=False, dynamic=False, options={"triton.cudagraphs": False})` for every call in fit and predict.
- `optimized`: `compiled` plus an Inductor post-grad pass that routes matched regions to the registered kernels (see [`docs/inference-kernel-interface.md`](../../docs/inference-kernel-interface.md)). SDM source is unchanged. By default these are the ops that are bitwise equal to compiled SDM plus the ICL attention kernel of `--attention`; `--ops` limits the pass to some named ops, and `--arms eager compiled` turns the kernels off.
- `optimized:<attention>`: like `optimized`, with its own ICL attention setting, compiled separately. Several of them are timed in one run, for example `--arms compiled optimized:off optimized:sm120 optimized:int8`.

Typical comparisons, as separate runs or as arms of one run (`compiled` is in every run):

| Optimized arm                       | Flags                               | Gate      |
| ----------------------------------- | ----------------------------------- | --------- |
| exact kernels only                  | `--attention off`                   | bitwise   |
| exact kernels + BF16 ICL attention  | (default)                           | tolerance |
| exact kernels + 8-bit ICL attention | `--attention int8` (or `int8+ends`) | opt-in    |
| BF16 ICL attention only             | `--ops icl_attention`               | tolerance |

Regions no kernel admits (other shapes, other model sizes) stay native. On a non-Blackwell GPU, or when no kernel admits any region, the optimized arm is skipped with a warning, the native arms are still measured, and the reason is in `summary.md` and `report.json` (`optimized_unavailable`). A failed kernel build is an error.

To use the kernels outside the benchmark, compile an SDM model with them in one call:

```python
from benchmark.inference import kernels

kernels.enable(model)  # torch.compile every task model with the kernels on
...
kernels.disable(model)
```

`kernels.enable(model, exact=True)` keeps only the kernels that are bitwise equal to compiled SDM end to end (ICL attention stays native). `kernels.enable(model, attention="int8")` (or `SDM_ICL_ATTENTION=int8`) opts in to 8-bit ICL attention.

## Supported scope

- **Contexts.** Compiled SDM runs any context: SDPA splits flash calls above 65,535 batch entries and attention and cell embedding chunk under `torch.compile`. The 100k runs here used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **What runs where.** Chunking splits the row blocks into context chunks and the column blocks into feature chunks once the context is above roughly 7,000 rows. The row-local kernels and the GEMM epilogues match every chunk; the column table kernels do not take a feature range, so at large contexts those regions stay native. See `docs/inference-kernels.md` (Coverage and limits).
- **Wider tables.** The default recipe allows up to 500 features. Beyond a kernel's checked size, those regions run native and the gain shrinks.
- **How the bounds were set.** Each exact kernel's `supports` admits rows up to the largest size checked bitwise against compiled native. `--shadow` re-checks every kernel on the real model.
- **ICL attention is not bitwise.** The BF16 `icl_attention` kernel is on by default and switches the gate to the tolerance rule below; `--attention off` (or `SDM_ICL_ATTENTION=off`) gates the run bitwise. Above 65,535 context rows, the largest size its accuracy was checked at, it stays native. 8-bit attention (`--attention int8`) is lossy, opt-in and has no row cap.

## What is timed

Outside the clock, before each call: `model.clear()`, `gc.collect()` and a fresh, seeded CUDA generator; Python garbage collection is disabled during the timed call (same for every arm). Inputs are already on the GPU. Inside the clock: CUDA synchronize, then `fit` + `predict` under BF16 autocast and `torch.inference_mode()`, then CUDA synchronize. Compilation and warmup are reported separately and never timed; a recompile during timing raises an error instead of being timed. Compiled code depends on the environment (torch, Triton, CUDA, GPU) and on Inductor's cache: a run that reuses cached compiled code (for example a repeat on the same machine) does not recompile, and a different environment can compile differently. The reported initial and repeat runs shared one Inductor cache. Arms are interleaved: each block runs an order of the arms followed by its reverse. When the blocks are a multiple of the number of permutations every permutation is used equally often; otherwise the orders are the rows of a Williams Latin square (each arm once in each position). The peak allocated and reserved GPU memory of each arm over its timed calls is in `report.json` (`peak_memory`) and `summary.md`. Every sample is kept.

## Correctness policy

Before timing, each arm runs once from identical state. The run captures the predicted probabilities and every tensor reachable through the fitted cache's dicts, lists and dataclasses. The preprocessing recipe object in the cache runs outside the compiled model and is not compared.

- `optimized` vs `compiled` must be bitwise equal on every tensor when every selected kernel is bitwise end to end (the default ops with `--attention off`). If they differ, nothing is timed (unless `--allow-mismatch` is passed), and the report names the first differing tensor.
- When a selected kernel is not bitwise (BF16 `icl_attention`), the gate becomes a tolerance gate: the optimized-vs-compiled maximum probability difference must not exceed the compiled-vs-eager one from the same run, and the predicted classes must all agree. Eager is run once for this, untimed when it is not an arm (`qualification.eager_timed`). `report.json` records which gate ran (`qualification.gate`).
- With a lossy opt-in kernel (`--attention int8` or `int8+ends`) the gate is `opt-in`: the distance to compiled (and compiled vs eager, with the eager arm) is recorded, the run is timed, and `summary.md` says it is lossy.
- `--shadow` is a per-kernel check: each kernel runs beside native on native's inputs and its outputs are compared bit for bit (`report.json`, `kernels.shadow`; status `shadow_bitwise` when all are equal).
- With `--qualify-only`, every arm is also profiled once and must launch the expected ICL attention kernel exactly once per fit call (none for eager and compiled; two fewer with `+ends`), or the status is `launch_verification_failed`; the counts are in `report.json` (`kernels.icl_attention_launches`).
- `compiled` vs `eager` is characterized, not gated: probability differences, argmax agreement, and the maximum difference per cache tensor.

## Outputs

- `samples.csv`: every timed call (`block`, `position`, `arm`, `seconds`).
- `report.json`: configuration, environment (SDM and benchmark git revisions, versions, GPU, precision flags, compile options), warmup times, qualification, statistics, and the kernel report.
- `summary.md`: a short human summary.

Statistics are per-arm mean/median/min/stdev. Each arm pair also gets the reduction in mean latency (in %), the per-block paired effects, the median block effect, and the number of blocks where the candidate was faster.

The PyTorch profiler prints `USDT ... profiler_start/stop` lines for the profiled calls; they are harmless, and no environment variable silences them.

The kernel report has CUDA kernel names, counts and times from one profiled call per arm. It also has the registry's selection decisions and the kernel launch counts for one optimized call. `kernels_verified` checks that every selected kernel shows its `cuda_symbols` in the optimized profile and not in the compiled one.

## Sweeps

```bash
python -m benchmark.inference.sweep run OUT            # Large, Medium, Small x 5k..100k; resumable
python -m benchmark.inference.sweep run OUT --dry-run  # list cells, arms and blocks
python -m benchmark.inference.sweep table OUT          # OUT/tables.md and OUT/table.csv
```

Each cell is one run with every arm interleaved: compiled, `optimized:off`, `optimized:sm120` (up to its 65,535-row cap), `optimized:int8`, eager at 5k and 50k, and at Large `int8+ends` (50k, 100k) and `sm120+uncapped` (100k), plus a repeat of Large 50k and 100k. Blocks are rounded up to a multiple of the cell's arm count. Runs pass `--allow-mismatch`, so a failed gate is timed and flagged in the table. A cell that runs out of memory is retried once with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, then recorded as OOM; failures go to `OUT/failures.log`. Each attempt keeps its benchmark output, log and an `nvidia-smi` sample (`smi.csv`); the table reports median power and SM clock over the timed calls. Arguments after `--` go to every run.

## Building kernels separately

```bash
python -m benchmark.inference.kernels.build --out DIR [--cutlass PATH]
```

Builds every module in `kernels/build.py` into `DIR/<module>.so`; modules whose sources and flags are unchanged are skipped.

## Adding a kernel

See [`docs/inference-kernel-interface.md`](../../docs/inference-kernel-interface.md).
