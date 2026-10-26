# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column query projection with query RMS fused (``linear_query_rms``).

Compiled SDM runs the column cross-attention query as addmm into a raw
[M, D] temporary, viewed per head, then SDM's Triton ``_rmsnorm_cast_kernel``
(no weight, eps 1e-6, channels = head). The raw query has no other reader, so
the kernel never stores it.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fx import Graph, Node

from benchmark.inference.kernels import fx_pass, registry, sm120
from benchmark.inference.kernels.fusions.kv_rms import (
    is_view,
    unweighted_rms_launch,
)

aten = torch.ops.aten
OP = "linear_query_rms"
MODULE = "query_rms_sm120"
EPS = 1e-6
# Largest M per D checked bitwise against the compiled addmm + norm.
MAX_M = {128: 33_029_640, 256: 20_600_904}
# (D, head) -> M band where cuBLAS uses another kernel (WMMA/split-K, other
# summation order). Bytes match cuBLAS elsewhere in 1..4096 and on a 7% log
# grid to 3.1M. Small's M = 128 is admitted; Medium's M = 256 is not.
M_GAP = {(256, 64): (1, 256), (128, 32): (934, 1024)}


def _reference(x: Tensor, weight: Tensor, bias: Tensor, head: int) -> Tensor:
    q = F.linear(x, weight, bias)
    return (
        F.rms_norm(q.float().unflatten(-1, (-1, head)), (head,), eps=EPS)
        .to(q.dtype)
        .flatten(-2)
    )


registry.define_op(
    OP,
    _reference,
    "per-head unweighted RMS (eps 1e-6) of linear(x, weight, bias)",
)


@torch.library.custom_op(
    f"sdm_inference::{OP}", mutates_args=(), tags=registry.EXACT_STRIDES
)
def linear_query_rms(
    x: Tensor, weight: Tensor, bias: Tensor, head: int, kernel: str
) -> Tensor:
    return registry.launch(registry.kernel(kernel), x, weight, bias, head)


@linear_query_rms.register_fake
def _(
    x: Tensor, weight: Tensor, bias: Tensor, head: int, kernel: str
) -> Tensor:
    return x.new_empty(x.shape)


def _rule(graph: Graph, norm: Node) -> bool:
    """Addmm(bias, x, permute(w)) -> views -> per-head rmsnorm; sole reader."""
    launch = unweighted_rms_launch(norm)
    if launch is None:
        return False
    view, out, head, eps = launch
    if not is_view(view) or eps != EPS or head != fx_pass.val(view).shape[-1]:
        return False
    user = norm
    while is_view(view) and fx_pass.only_user(view, user):
        view, user = view.args[0], view
    addmm = view
    if not (fx_pass.plain_addmm(addmm) and fx_pass.only_user(addmm, user)):
        return False
    bias, x, transposed = addmm.args
    if not fx_pass.is_call(transposed, aten.permute.default, [1, 0]):
        return False
    weight = transposed.args[0]
    kernel = registry.select(
        OP, fx_pass.val(x), fx_pass.val(weight), fx_pass.val(bias), head
    )
    if kernel is None:
        return False
    with graph.inserting_before(out):
        fused = graph.call_function(
            torch.ops.sdm_inference.linear_query_rms.default,
            (x, weight, bias, head, kernel.name),
        )
        fused.meta["val"] = fx_pass.val(addmm).new_empty(
            fx_pass.val(addmm).shape
        )
        shaped = graph.call_function(
            aten.view.default, (fused, list(fx_pass.val(out).shape))
        )
        shaped.meta["val"] = fx_pass.val(out)
    fx_pass.replace(out, shaped)
    return True


fx_pass.add_rule(fx_pass.TRITON_KERNEL, OP, _rule)


def _supports(
    x: Tensor, weight: Tensor, bias: Tensor, head: int
) -> str | None:
    if x.dim() != 2 or (x.size(1), head) not in M_GAP:
        return f"needs x [M, D] with (D, head) in {sorted(M_GAP)}"
    m, d = x.shape
    lo, hi = M_GAP[d, head]
    if not 1 <= m <= MAX_M[d]:
        return f"needs M in [1, {MAX_M[d]}]; got M={m}"
    if lo <= m <= hi:
        return f"M in [{lo}, {hi}] differs from cuBLAS; got M={m}"
    if weight.shape != (d, d) or bias.shape != (d,):
        return f"needs a {d}x{d} weight and bias"
    return sm120.policy(x, weight, bias)


def register(directory: Path) -> None:
    module = sm120.load_module(directory, MODULE)

    def impl(x: Tensor, weight: Tensor, bias: Tensor, head: int) -> Tensor:
        out = torch.empty_like(x)
        module.run(x, weight, bias, out, head)
        return out

    registry.register(
        registry.Kernel(
            op=OP,
            name="sm120_linear_query_rms",
            supports=_supports,
            impl=impl,
            cuda_symbols=("9query_rms5Query",),
        )
    )
