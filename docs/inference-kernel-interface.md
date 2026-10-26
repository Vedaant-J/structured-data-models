# Inference kernel interface

How a kernel plugs into the KumoTabular fit + predict benchmark (`benchmark/inference/`). What each kernel does is in [`inference-kernels.md`](inference-kernels.md).

## Three pieces

1. **Op**: an operation with pure-PyTorch reference semantics, declared with `registry.define_op`, for example `linear_gelu(x, weight, bias) = gelu(linear(x, weight, bias))` with exact (erf) GELU. Each op is also a `torch.library` custom op `sdm_inference::<op>` whose last argument is the name of the selected kernel.
2. **Kernel** (`registry.Kernel`): one implementation of one op.
   - `op`, `name`: the op it implements and a unique name, e.g. `sm120_linear_gelu_k256_n512`.
   - `supports(*args) -> None | str`: inspects tensor metadata only (shape, dtype, stride, storage offset, device, compute capability) and global policy (precision flags). Returns `None` to admit, else a short reason. Never launches work or synchronizes. It cannot check inference mode, because compilation runs outside it; the benchmark runs every call under `torch.inference_mode()`.
   - `impl(*args) -> Tensor | tuple[Tensor, ...] | None`: allocates and returns new outputs. It mutates no argument.
   - `cuda_symbols`: substrings of the CUDA kernel names `impl` launches, so a profile can show it ran.
   - `bitwise(*args) -> bool` (optional, default always true): whether the args lie where the kernel was checked bitwise against native. `supports` may admit sizes above that, up to a hard limit, where the kernel stays within a measured tolerance; such runs use the benchmark's tolerance gate.
   - `exact` (default `True`): `False` marks a lossy kernel that is selectable only through an opt-in setting (the 8-bit ICL attention); the benchmark then records its distance instead of gating on it.
3. **Graph rule**, run by the Inductor post-grad pass in `benchmark/inference/kernels/fx_pass.py`, which is installed only in the optimized arm's `torch.compile` options. A rule matches its op's pre-fusion ATen form, calls `registry.select(op, *fake_args)` on the compile-time metadata, and rewrites the region to `sdm_inference::<op>(..., kernel=name)` only if a kernel admits it. Unmatched or rejected regions stay native and Inductor compiles them as usual. SDM source is not modified. The fallback is chosen at compile time, before any launch, and is recorded in the census with the rejection reasons.

All three pieces of a kernel family live in one file, `benchmark/inference/kernels/fusions/<family>.py`: the op's reference and `define_op`, its custom op and fake implementation, its rule (`fx_pass.add_rule`) and `register(directory)`, which loads the family's built CUDA modules from `directory` and registers its kernels.

## The row table

SDM's row embedding keeps a table `[C, F+4, D]` (4 readout tokens plus `F` cell features per context row) and updates it in place. The whole forward is one compiled graph, so after functionalization the table is an in-graph value: `torch.empty`, then one new value per update (`slice_scatter` into `t[:, 4:]` for the column blocks, `t + x` for the row residuals). `column_projection_rms` finds the table this way (`_table_value` and `_table_rows` in `fusions/projection_rms.py`); it also accepts the table as a flat graph input, which is how the unit tests build it.

No op writes the table. Inductor decides per graph which values it stores: native SDM never stores the table after a column block, its norms read Inductor's private copies, and the row residual recomputes the table in FP32. A custom op that takes the table makes Inductor store it as BF16, and later readers load the rounded value, so the model is no longer bitwise end to end even though each kernel is (see [No row-table ops](inference-kernels.md#how-the-kernels-get-into-the-model)). A new kernel that reads an in-graph value with other native readers must keep native's storage decisions; the bitwise end-to-end gate checks this.

## Contract a kernel must meet

- **Exact against compiled native.** (The exceptions are `icl_attention`, below, and the lossy opt-in kernels.) On every input `supports` admits, the output must be bitwise equal to what clean SDM compiled with Inductor produces for that region, under the benchmark settings: BF16 autocast, inference mode, TF32 off, reduced-precision BF16 reductions off. Compiled native keeps FP32 between fused pointwise ops and rounds once at each stored buffer, so match that, not eager SDM (which rounds after every op). `--shadow` checks this on the real model: every kernel runs beside native on native's inputs and each output is compared bit for bit.
- **Honest `supports`.** Reject anything the kernel was not qualified for (shape, layout, M range, GPU arch). The benchmark gate checks whole-model outputs, so an over-broad `supports` shows up as a mismatch, not a crash.
- **No hidden state.** No caching across calls, no CUDA graphs, no stream changes; launch on the current stream.
- **Exact strides.** Custom ops are declared with `tags=registry.EXACT_STRIDES` (`torch.Tag.needs_exact_strides`). Otherwise Inductor picks its own layout for in-graph buffers, for example a feature-major table, and hands it to the kernel.

## Adding a kernel

```python
from benchmark.inference.kernels import registry, sm120

def register(directory):
    module = sm120.load_module(directory, "my_family_sm120")
    registry.register(registry.Kernel(op="linear_gelu", name="sm120_my_kernel",
                                      supports=supports, impl=impl, cuda_symbols=("my_namespace",)))
```

`integration.install` builds the CUDA modules with `build.build` and passes the output directory to `fusions.load`, which imports every module in `fusions/` in name order and calls its `register`. Earlier registrations win. `sm120.py` holds the shared helpers: `load_module`, `policy` (CUDA, SM 12.0, BF16, contiguous, TF32 and reduced-precision reductions off), `MIN_M` and `EPS`.

A new op needs a reference in `registry.define_op`, a custom op with a fake implementation, and a rule registered with `fx_pass.add_rule(target, op, rule)`; rules for one target are tried in order until one returns `True`. `fx_pass.triton_kernel_name(node)` names the user Triton kernel behind a `triton_kernel_wrapper_functional` node, e.g. SDM's `_rmsnorm_cast_kernel`; `fusions/kv_rms.py` has `unweighted_rms_launch` for SDM's unweighted per-head norm.

CUDA sources live in `benchmark/inference/kernels/csrc/<module>/`, one directory per built module, listed in `build.MODULES` with their flags; headers shared by several modules are in `csrc/common/`, and `csrc/.clang-format` sets the style. The optimized arm builds missing or stale modules on first use (`--kernels DIR`, default `$SDM_KERNEL_CACHE` or `~/.cache/sdm-inference-kernels/<torch>-<cuda>`); `python -m benchmark.inference.kernels.build --out DIR` builds ahead of time. `--ops a b ...` restricts the optimized arm to some ops, for cumulative measurements.

## What the benchmark reports

- `decisions`: every compile-time selection (op, kernel or `native`, rejection reasons, shapes).
- `launches_per_optimized_call`: runtime launch count per kernel.
- `verification`: each selected kernel's `cuda_symbols` present in the optimized profile and absent from the compiled one.
- Qualification: optimized vs compiled bitwise on the probabilities and every tensor reachable through the fitted cache's dicts, lists and dataclasses, before any timing. The preprocessing recipe object in the cache runs outside the compiled model and is not compared.

## Ops

Every op is a `sdm_inference::<op>` custom op with a reference in `registry.OPS`; the docstring of its file in `benchmark/inference/kernels/fusions/` gives its exact semantics. CUDA modules are under `benchmark/inference/kernels/csrc/`.

| Op                                   | Python                                              | CUDA                                                                                     | What it fuses                                                                                                                                                                                                  |
| ------------------------------------ | --------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `linear_gelu`                        | `linear_gelu.py`; first gating layer in `gating.py` | `linear_gelu_sm120/linear_gelu.cu`, `gating_sm120/gating.cu`                             | linear + exact GELU (embedding MLPs, first gating layer, dataset FFN)                                                                                                                                          |
| `gated_log_scale`                    | `gating.py`                                         | `gating_sm120/gating.cu`                                                                 | second gating linear + tanh + log-scale query multiply                                                                                                                                                         |
| `rope_rms`, `rope_rms_uneven`        | `rope_rms.py`                                       | `rope_rms_sm120/rope_rms.cu`                                                             | split-half RoPE + per-head Q/K RMSNorm; `rope_rms_uneven` is the last row layer (4 readout queries), with Q and K of different lengths                                                                         |
| `linear_kv_key_rms`                  | `kv_rms.py`                                         | `kv_rms_sm120/kv_rms.cu`                                                                 | KV projection + per-head key RMSNorm                                                                                                                                                                           |
| `linear_query_rms`                   | `query_rms.py`                                      | `query_rms_sm120/query_rms.cu`                                                           | query projection + per-head RMSNorm                                                                                                                                                                            |
| `column_projection_rms`              | `projection_rms.py`                                 | `projection_rms_sm120/projection_rms.cu`                                                 | column output projection + residual + RMSNorm                                                                                                                                                                  |
| `icl_query_rms_scale`, `icl_key_rms` | `icl_qk.py`                                         | `icl_qk_sm120/icl_qk.cu`                                                                 | ICL query per-head RMSNorm + LogScale multiply; ICL key RMSNorm read in place from the packed KV projection, plus contiguous K/V cache heads (the last fit layer's KV projection narrowed to the cached heads) |
| `icl_attention`                      | `icl_attention.py`                                  | `icl_attention_sm120/icl_attention.cu`, `icl_attention_int8_sm120/icl_attention_int8.cu` | not a fusion and **not bitwise**: ICL fit self-attention in a custom SM120 BF16 kernel, FlashAttention 2, or the lossy opt-in INT8 QK / FP8 PV kernel; see the setting below                                   |

## ICL attention setting

`$SDM_ICL_ATTENTION`, `--attention`, or `kernels.enable(model, attention=...)` (which overrides the environment through `registry.enabled(attention=...)`) picks the `icl_attention` kernel:

| Setting           | Kernel                                                      | Gate                                |
| ----------------- | ----------------------------------------------------------- | ----------------------------------- |
| `sm120` (default) | `sm120_icl_attention_d64`, BF16, at most 65,535 rows        | tolerance                           |
| `sm120+uncapped`  | the same kernel without the row cap (diagnostic)            | tolerance                           |
| `fa2`             | `fa2_icl_attention_d64` (needs `flash_attn` for SM 12.0)    | tolerance                           |
| `int8`            | `sm120_int8_icl`: INT8 Q·Kᵀ with K smoothing, FP8 P·V       | `opt-in` (distance recorded, timed) |
| `int8+ends`       | as `int8`, first and last ICL fit call of each graph native | `opt-in`                            |
| `off`             | none                                                        | bitwise (with the exact ops)        |

The rule matches the flash node of a self-attention with head dim 64 and at least 1,024 rows (fit; predict has 16 queries against the cache). The BF16 kernels admit at most 65,535 rows, the largest context at which their error and determinism were checked; the INT8 kernel has no row cap but needs batch 1 and 32-bit element offsets. `exact=True` keeps attention native whatever the setting.

## Shape bounds and scope

Exactness against cuBLAS depends on the algorithm cuBLAS picks per shape. Each kernel's `supports` admits rows only up to the largest size that was checked bitwise. `sm120.MIN_M = 1024` is the lower bound of the GEMM kernels; smaller (query-side) calls stay native, and some kernels have per-shape lower bounds.

SDM chunks attention and cell embedding under `torch.compile` (a memory budget of `SDM_CHUNK_MEMORY_FRACTION`, default 5% of the GPU, per chunk). The row blocks run on chunks of about 7,200 contexts (Large and Medium; about 14,400 for Small; 57 table rows) and the column blocks on chunks of a few features once the context is above about 7,500 rows, so no upper bound of an exact kernel binds at 100k contexts. The ICL block is not chunked: its ops see M = C. `icl_key_rms` narrows the last fit layer's KV projection only for M in 513..65,535 (checked bitwise); above that it runs the full projection. Rules match the row-chunked row blocks (`rope_rms`, gating, the GEMM epilogues). Feature-chunked column blocks read a feature range of the table, which `column_projection_rms` does not take, so those regions stay native at large contexts. Re-check exactness before widening any bound; `--shadow` does that on the real model.

## Testing a kernel

The kernel tests are in `test/benchmark/inference/` (`test_fusions.py`, `test_icl_attention.py`, `test_icl_qk.py`). `conftest.py` provides the fixture `matches_native(fn, make_args, kernel)`: on an SM 12.0 GPU it builds and registers every kernel, compiles a small function reproducing the SDM pattern twice (plain, and with the kernel pass), and asserts that `kernel` was selected and launched and that every output and every input is bitwise equal. Off SM 12.0 the tests are skipped. For a kernel that is not bitwise, `close_to_native(fn, make_args, kernel, atol)` asserts the same within a maximum absolute difference. `stays_native(fn, make_args, op)` asserts that the rule ran but chose no kernel, `skips_rule` that the rule left the function alone, and `rejects(kernel, *args)` that `supports` refuses the arguments.

On the real model, `python -m benchmark.inference --shadow ...` (also `fx_pass.compile_options(shadow=True)`) keeps native on the data path and routes each rewritten output through `sdm_inference::shadow_compare` (in `fx_pass.py`), which records per op how many outputs were bitwise equal (`report.json`, `kernels.shadow`).

## Using the kernels in SDM code

```python
from benchmark.inference import kernels

kernels.enable(model)                      # bitwise ops + BF16 ICL attention
kernels.enable(model, exact=True)          # only ops bitwise equal to compiled SDM
kernels.enable(model, attention="int8")    # opt-in 8-bit ICL attention
kernels.disable(model)
```

`kernels.compile_with_kernels(fn)` does the same for one function; the benchmark's optimized arm uses it.
