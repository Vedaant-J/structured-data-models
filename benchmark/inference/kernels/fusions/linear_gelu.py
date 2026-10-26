# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``linear_gelu`` for the embedding MLPs and the dataset FFN.

The first gating layer is also a ``linear_gelu``; its kernel is in
``gating.py``.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fx import Graph, Node

from benchmark.inference.kernels import fx_pass, registry, sm120
from benchmark.inference.kernels.fx_pass import is_call, only_user, val

aten, prims = torch.ops.aten, torch.ops.prims

MODULE = "linear_gelu_sm120"
# (K, N) -> largest M checked bitwise against compiled native.
MAX_M = {(128, 256): 33_029_640, (256, 512): 8_388_608, (1024, 2048): 500_000}
# Mangled names: linear_gelu::BiasGelu, linear_gelu::GeluEpilogue (K1024).
SYMBOLS = ("11linear_gelu8BiasGelu", "11linear_gelu12GeluEpilogue")


def _reference(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    return F.gelu(F.linear(x, weight, bias))


registry.define_op(
    name="linear_gelu",
    reference=_reference,
    doc="gelu(linear(x, weight, bias)) with exact (erf) GELU",
)


@torch.library.custom_op(
    "sdm_inference::linear_gelu", mutates_args=(), tags=registry.EXACT_STRIDES
)
def linear_gelu(
    x: Tensor, weight: Tensor, bias: Tensor, kernel: str
) -> Tensor:
    return registry.launch(registry.kernel(kernel), x, weight, bias)


@linear_gelu.register_fake
def _(x: Tensor, weight: Tensor, bias: Tensor, kernel: str) -> Tensor:
    return x.new_empty(x.size(0), weight.size(0))


def _rule(graph: Graph, erf: Node) -> bool:
    """Addmm -> [view] -> f32 -> 0.5x(1 + erf(x/sqrt2)) -> bf16."""
    scaled = erf.args[0]
    if not is_call(scaled, aten.mul.Tensor, 0.7071067811865476):
        return False
    upcast = scaled.args[0]
    if not is_call(upcast, prims.convert_element_type.default, torch.float32):
        return False
    source = upcast.args[0]
    view = (
        source
        if is_call(source, aten.view.default)
        or is_call(source, aten.reshape.default)
        else None
    )
    addmm = view.args[0] if view is not None else source
    if not fx_pass.plain_addmm(addmm):
        return False
    bias, x, transposed = addmm.args
    if not is_call(transposed, aten.permute.default, [1, 0]):
        return False
    half = [u for u in upcast.users if is_call(u, aten.mul.Tensor, 0.5)]
    shifted = [u for u in erf.users if is_call(u, aten.add.Tensor, 1)]
    if (
        len(half) != 1
        or len(shifted) != 1
        or set(upcast.users)
        != {
            scaled,
            half[0],
        }
    ):
        return False
    product = [u for u in half[0].users if u.target == aten.mul.Tensor]
    if len(product) != 1 or set(product[0].args) != {half[0], shifted[0]}:
        return False
    downcast = list(product[0].users)
    if len(downcast) != 1 or not is_call(
        downcast[0], prims.convert_element_type.default, torch.bfloat16
    ):
        return False
    if not (
        only_user(addmm, view or upcast)
        and (view is None or only_user(view, upcast))
        and only_user(scaled, erf)
        and only_user(erf, shifted[0])
        and only_user(shifted[0], product[0])
        and only_user(half[0], product[0])
    ):
        return False
    weight = transposed.args[0]
    kernel = registry.select("linear_gelu", val(x), val(weight), val(bias))
    if kernel is None:
        return False
    with graph.inserting_before(downcast[0]):
        fused = graph.call_function(
            torch.ops.sdm_inference.linear_gelu.default,
            (x, weight, bias, kernel.name),
        )
        fused.meta["val"] = val(addmm)
        if view is not None:
            fused = graph.call_function(view.target, (fused, view.args[1]))
            fused.meta["val"] = val(view)
    fx_pass.replace(downcast[0], fused)
    return True


fx_pass.add_rule(aten.erf.default, "linear_gelu", _rule)


def _kernel(module, k: int, n: int, max_m: int) -> registry.Kernel:
    def supports(x: Tensor, weight: Tensor, bias: Tensor) -> str | None:
        if (
            x.dim() != 2
            or x.size(1) != k
            or not sm120.MIN_M <= x.size(0) <= max_m
        ):
            return f"needs x of shape [{sm120.MIN_M}<=M<={max_m}, {k}]"
        if weight.shape != (n, k) or bias.shape != (n,):
            return f"needs a {n}x{k} weight and bias"
        return sm120.policy(x, weight, bias)

    def impl(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        out = x.new_empty(x.size(0), n)
        module.run(x, weight, bias, out)
        return out

    return registry.Kernel(
        op="linear_gelu",
        name=f"sm120_linear_gelu_k{k}_n{n}",
        supports=supports,
        impl=impl,
        cuda_symbols=(SYMBOLS[k == 1024],),
    )


def register(directory: Path) -> None:
    module = sm120.load_module(directory, MODULE)
    for (k, n), max_m in MAX_M.items():
        registry.register(_kernel(module, k, n, max_m))
