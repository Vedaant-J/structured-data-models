# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column KV projection with per-head key RMS fused (``linear_kv_key_rms``).

Compiled SDM runs the column cross-attention KV as addmm into packed
[M, 2D], a split into K | V, a contiguous copy of K and SDM's Triton
``_rmsnorm_cast_kernel`` on it (no weight, eps 1e-6, channels = head). When
that copy is raw K's only reader, the kernel never stores raw K. V stays in
the packed buffer, so its readers are unchanged.
"""

import operator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch.fx import Graph, Node

from benchmark.inference.kernels import fx_pass, registry, sm120

aten = torch.ops.aten
OP = "linear_kv_key_rms"
MODULE = "kv_rms_sm120"
EPS = 1e-6
# Largest M per D checked bitwise against the compiled addmm + norm.
MAX_M = {128: 19_942_176, 256: 12_040_400}
# (D, head) -> smallest M whose bytes match cuBLAS addmm; at D=256 cuBLAS
# sums in another order for 17 <= M <= 128.
MIN_M = {(256, 64): 129, (128, 32): 1}


def _reference(
    x: Tensor, weight: Tensor, bias: Tensor, head: int
) -> tuple[Tensor, Tensor]:
    k, v = F.linear(x, weight, bias).chunk(2, -1)
    k = F.rms_norm(k.float().unflatten(-1, (-1, head)), (head,), eps=EPS)
    return k.to(v.dtype).flatten(-2), v


registry.define_op(
    OP,
    _reference,
    "k, v = linear(x, w, b).chunk(2); per-head unweighted RMS (eps 1e-6) of k",
)


@torch.library.custom_op(
    f"sdm_inference::{OP}", mutates_args=(), tags=registry.EXACT_STRIDES
)
def linear_kv_key_rms(
    x: Tensor, weight: Tensor, bias: Tensor, head: int, kernel: str
) -> tuple[Tensor, Tensor]:
    return registry.launch(registry.kernel(kernel), x, weight, bias, head)


@linear_kv_key_rms.register_fake
def _(
    x: Tensor, weight: Tensor, bias: Tensor, head: int, kernel: str
) -> tuple[Tensor, Tensor]:
    m, d = x.shape
    return x.new_empty(m, d), x.new_empty(m, 2 * d)[:, d:]


def is_view(node: object) -> bool:
    return fx_pass.is_call(node, aten.view.default) or fx_pass.is_call(
        node, aten.reshape.default
    )


def unweighted_rms_launch(norm: Node) -> tuple[Node, Node, int, float] | None:
    """SDM ``rmsnorm_cast`` without weight or RoPE: (x, out, channels, eps)."""
    if fx_pass.triton_kernel_name(norm) != "_rmsnorm_cast_kernel":
        return None
    tensors = norm.kwargs["kwargs"]
    const = kernel_side_table.get_constant_args(
        norm.kwargs["constant_args_idx"]
    )
    outs = list(norm.users)
    if (
        set(tensors) != {"x_ptr", "out_ptr"}
        or any(
            const.get(k, 0) is not None
            for k in ("weight_ptr", "cos_ptr", "sin_ptr")
        )
        or not fx_pass.only_user(tensors["out_ptr"], norm)
        or len(outs) != 1
        or outs[0].args[1:] != ("out_ptr",)
        or fx_pass.val(outs[0]).dtype != fx_pass.val(tensors["x_ptr"]).dtype
    ):
        return None
    return tensors["x_ptr"], outs[0], const.get("channels"), const.get("eps")


def _rule(graph: Graph, norm: Node) -> bool:
    """Addmm -> [reshape] -> split(D); K -> reshape -> clone -> rmsnorm; V any.

    Raw K (split output 0) must have no reader other than that copy.
    """
    launch = unweighted_rms_launch(norm)
    if launch is None:
        return False
    clone, out, head, eps = launch
    if not (
        fx_pass.is_call(clone, aten.clone.default)
        and fx_pass.only_user(clone, norm)
    ):
        return False
    shaped = clone.args[0]
    if not (is_view(shaped) and fx_pass.only_user(shaped, clone)):
        return False
    raw_k = shaped.args[0]
    if not (
        fx_pass.is_call(raw_k, operator.getitem)
        and raw_k.args[1] == 0
        and fx_pass.only_user(raw_k, shaped)
    ):
        return False
    split = raw_k.args[0]
    if not fx_pass.is_call(
        split, aten.split.Tensor, fx_pass.val(raw_k).shape[-1], -1
    ):
        return False
    halves = {u.args[1]: u for u in split.users}
    if len(split.users) != 2 or set(halves) != {0, 1}:
        return False
    source = split.args[0]
    addmm = source.args[0] if is_view(source) else source
    if not (
        fx_pass.plain_addmm(addmm)
        and fx_pass.only_user(source, split)
        and (addmm is source or fx_pass.only_user(addmm, source))
    ):
        return False
    bias, x, transposed = addmm.args
    if not (
        fx_pass.is_call(transposed, aten.permute.default, [1, 0])
        and fx_pass.val(raw_k).shape[-1] == fx_pass.val(x).shape[1]
        and eps == EPS
        and head == fx_pass.val(shaped).shape[-1]
    ):
        return False
    weight = transposed.args[0]
    kernel = registry.select(
        OP, fx_pass.val(x), fx_pass.val(weight), fx_pass.val(bias), head
    )
    if kernel is None:
        return False
    m, d = fx_pass.val(x).shape
    with graph.inserting_before(out):
        fused = graph.call_function(
            torch.ops.sdm_inference.linear_kv_key_rms.default,
            (x, weight, bias, head, kernel.name),
        )
        packed = fx_pass.val(addmm)
        fused.meta["val"] = (packed.new_empty(m, d), packed[:, d:])
        replacements = []
        for i, old in ((0, out), (1, halves[1])):
            part = graph.call_function(operator.getitem, (fused, i))
            part.meta["val"] = fused.meta["val"][i]
            view = graph.call_function(
                aten.view.default, (part, list(fx_pass.val(old).shape))
            )
            view.meta["val"] = part.meta["val"].view(fx_pass.val(old).shape)
            replacements.append((old, view))
    for old, new in replacements:
        fx_pass.replace(old, new)
    return True


fx_pass.add_rule(fx_pass.TRITON_KERNEL, OP, _rule)


def _supports(
    x: Tensor, weight: Tensor, bias: Tensor, head: int
) -> str | None:
    if x.dim() != 2 or (x.size(1), head) not in MIN_M:
        return f"needs x [M, D] with (D, head) in {sorted(MIN_M)}"
    m, d = x.shape
    if not MIN_M[d, head] <= m <= MAX_M[d]:
        return f"needs {MIN_M[d, head]} <= M <= {MAX_M[d]}"
    if weight.shape != (2 * d, d) or bias.shape != (2 * d,):
        return f"needs a {2 * d}x{d} weight and bias"
    return sm120.policy(x, weight, bias)


def register(directory: Path) -> None:
    module = sm120.load_module(directory, MODULE)

    def impl(
        x: Tensor, weight: Tensor, bias: Tensor, head: int
    ) -> tuple[Tensor, Tensor]:
        m, d = x.shape
        packed, key = x.new_empty(m, 2 * d), x.new_empty(m, d)
        module.run(x, weight, bias, packed, key, head)
        return key, packed[:, d:]

    # kv_rms::Key mangles to 6kv_rms3Key in profiler symbols.
    registry.register(
        registry.Kernel(
            op=OP,
            name="sm120_linear_kv_key_rms",
            supports=_supports,
            impl=impl,
            cuda_symbols=("6kv_rms3Key",),
        )
    )
