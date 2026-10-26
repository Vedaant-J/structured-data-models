# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate MLP of the row block's ``GatedLogScale`` query scaling.

- First layer: ``linear_gelu`` with K = head (32 or 64), N = 64.
- Second layer: ``gated_log_scale``, the second linear plus the log-scale/tanh
  query multiply, as one GEMM with the gate in its epilogue.

Both match cuBLAS's wmma 32x32_64x1 kernel plus Inductor's pointwise kernel
bitwise.
"""

from pathlib import Path

import torch
from torch import Tensor
from torch.fx import Graph, Node

from benchmark.inference.kernels import fx_pass, registry, sm120
from benchmark.inference.kernels.fusions.kv_rms import is_view
from benchmark.inference.kernels.fx_pass import is_call, val

aten, prims = torch.ops.aten, torch.ops.prims
MODULE = "gating_sm120"
HEADS = (32, 64)
HIDDEN = 64
MIN_M = sm120.MIN_M
# Head width -> largest M (query rows) checked bitwise against compiled native.
FIRST_MAX_M = {32: 132_118_560, 64: 40_000_000}
SECOND_MAX_M = {32: 80_937_920, 64: 49_583_864}


def _reference(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    query: Tensor,
    head_scale: Tensor,
    logscale: float,
) -> Tensor:
    z = torch.addmm(bias, x, weight.t()).float()
    scale = head_scale.float().repeat(query.size(0) // 4 + 1)[: query.size(0)]
    q = query.float() * logscale * scale[:, None]
    return (q * (torch.tanh(z) + 1)).to(query.dtype)


registry.define_op(
    name="gated_log_scale",
    reference=_reference,
    doc="query * logscale * head_scale[row % 4] * "
    "(1 + tanh(linear(x, weight, bias)))",
)


@torch.library.custom_op(
    "sdm_inference::gated_log_scale",
    mutates_args=(),
    tags=registry.EXACT_STRIDES,
)
def gated_log_scale(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    query: Tensor,
    head_scale: Tensor,
    logscale: float,
    kernel: str,
) -> Tensor:
    return registry.launch(
        registry.kernel(kernel), x, weight, bias, query, head_scale, logscale
    )


@gated_log_scale.register_fake
def _(x, weight, bias, query, head_scale, logscale, kernel):
    return query.new_empty(query.shape)


def _operands(mul: Node, test) -> tuple[Node, Node] | None:
    """(matching, other) operands of a binary node, in either order."""
    a, b = mul.args[:2]
    return (a, b) if test(a) else (b, a) if test(b) else None


def _gated_log_scale(graph: Graph, tanh: Node) -> bool:
    """Match ``(q * logscale * head_scale) * (1 + tanh(addmm(...)))``."""
    shaped = tanh.args[0]
    if not is_view(shaped):
        return False
    addmm = shaped.args[0]
    if not fx_pass.plain_addmm(addmm):
        return False
    bias, x, transposed = addmm.args
    if not is_call(transposed, aten.permute.default, [1, 0]):
        return False
    users = list(tanh.users)
    if len(users) != 1 or not is_call(users[0], aten.add.Tensor, 1):
        return False
    shifted = users[0]
    if len(shifted.users) != 1 or not is_call(
        final := next(iter(shifted.users)), aten.mul.Tensor
    ):
        return False
    pair = _operands(final, lambda n: n is shifted)
    scaled = pair and pair[1]
    if not is_call(scaled, aten.mul.Tensor) or len(scaled.users) != 1:
        return False
    pair = _operands(scaled, is_view)
    if pair is None:
        return False
    head_view, logscaled = pair
    convert = head_view.args[0]
    if not (
        is_call(convert, prims.convert_element_type.default, torch.bfloat16)
        and is_call(logscaled, aten.mul.Tensor)
        and len(logscaled.users) == 1
    ):
        return False
    pair = _operands(logscaled, lambda n: is_call(n, aten.full.default))
    if pair is None:
        return False
    full, query = pair
    head_scale = convert.args[0]
    # The kernel scales row r by head_scale[r % 4]: heads on dim -2.
    heads = [1] * (val(query).dim() - 2) + [4, 1]
    if not (
        isinstance(full.args[1], float)
        and val(full).numel() == 1
        and val(head_scale).dtype == torch.float32
        and val(head_scale).dim() == 1
        and list(val(head_view).shape) == heads
        and val(query).shape[-2] == 4
        and len(addmm.users) == 1
        and len(shaped.users) == 1
    ):
        return False
    head = val(query).size(-1)
    if (
        val(query).shape != val(shaped).shape
        or val(final).shape != val(query).shape
    ):
        return False
    q2 = val(query).reshape(-1, head)
    weight = transposed.args[0]
    logscale = float(full.args[1])
    kernel = registry.select(
        "gated_log_scale",
        val(x),
        val(weight),
        val(bias),
        q2,
        val(head_scale),
        logscale,
    )
    if kernel is None:
        return False
    with graph.inserting_before(final):
        flat = graph.call_function(aten.reshape.default, (query, [-1, head]))
        flat.meta["val"] = q2
        fused = graph.call_function(
            torch.ops.sdm_inference.gated_log_scale.default,
            (x, weight, bias, flat, head_scale, logscale, kernel.name),
        )
        fused.meta["val"] = q2.new_empty(q2.shape)
        out = graph.call_function(
            aten.reshape.default, (fused, list(val(final).shape))
        )
        out.meta["val"] = val(final)
    fx_pass.replace(final, out)
    return True


fx_pass.add_rule(aten.tanh.default, "gated_log_scale", _gated_log_scale)


def _first(module, head: int) -> registry.Kernel:
    def supports(x: Tensor, weight: Tensor, bias: Tensor) -> str | None:
        if (
            x.dim() != 2
            or x.size(1) != head
            or not MIN_M <= x.size(0) <= FIRST_MAX_M[head]
        ):
            return (
                f"needs x of shape [{MIN_M}<=M<={FIRST_MAX_M[head]}, {head}]"
            )
        if weight.shape != (HIDDEN, head) or bias.shape != (HIDDEN,):
            return f"needs a {HIDDEN}x{head} weight and bias"
        return sm120.policy(x, weight, bias)

    def impl(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        out = x.new_empty(x.size(0), HIDDEN)
        module.first(x, weight, bias, out)
        return out

    return registry.Kernel(
        op="linear_gelu",
        name=f"sm120_linear_gelu_k{head}_n{HIDDEN}",
        supports=supports,
        impl=impl,
        cuda_symbols=("6gating8BiasGelu",),
    )


def _second_supports(
    x, weight, bias, query, head_scale, logscale
) -> str | None:
    if query.dim() != 2 or query.size(1) not in HEADS:
        return f"needs query [M, head] with head in {HEADS}"
    m, head = query.shape
    if not MIN_M <= m <= SECOND_MAX_M[head]:
        return f"needs {MIN_M}<=M<={SECOND_MAX_M[head]}"
    if (
        x.shape != (m, HIDDEN)
        or weight.shape != (head, HIDDEN)
        or bias.shape != (head,)
    ):
        return f"needs x [M, {HIDDEN}], a {head}x{HIDDEN} weight and bias"
    if head_scale.shape != (4,) or head_scale.dtype != torch.float32:
        return "needs a float32 head scale of 4 heads"
    if head_scale.device != query.device or not head_scale.is_contiguous():
        return "needs a contiguous head scale on the query device"
    return sm120.policy(x, weight, bias, query)


def register(directory: Path) -> None:
    module = sm120.load_module(directory, MODULE)

    def second(x, weight, bias, query, head_scale, logscale):
        out = torch.empty_like(query)
        module.second(x, weight, bias, query, head_scale, out, logscale)
        return out

    for head in HEADS:
        registry.register(_first(module, head))
    registry.register(
        registry.Kernel(
            op="gated_log_scale",
            name="sm120_gated_log_scale",
            supports=_second_supports,
            impl=second,
            cuda_symbols=("6gating12GateEpilogue",),
        )
    )
