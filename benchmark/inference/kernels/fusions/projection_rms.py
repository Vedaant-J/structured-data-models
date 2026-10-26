# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column attention out-projection + residual + weighted RMSNorm (D 128/256).

``(raw, rms_norm(RN(raw + table[:, 4:])))`` as ``[F, C, D]``.
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
MODULE = "projection_rms_sm120"
READOUTS = 4
WIDTHS = (128, 256)
# Largest M = C * F per D checked bitwise against the compiled addmm + norm.
MAX_M = {128: 19_815_464, 256: 11_982_968}


def _norm(x: Tensor, norm_weight: Tensor, eps: float) -> Tensor:
    return F.rms_norm(x.float(), (x.size(-1),), norm_weight, eps).bfloat16()


def _column_reference(x, weight, bias, table, norm_weight, eps):
    c, t, d = table.shape
    f = x.size(0) // c
    raw = torch.addmm(bias, x, weight.t())
    residual = (
        raw.view(f, c, d).float() + table[:, t - f :].transpose(0, 1).float()
    ).bfloat16()
    return raw, _norm(residual, norm_weight, eps)


registry.define_op(
    "column_projection_rms",
    _column_reference,
    "(raw, rms_norm(raw + table[:, 4:] as [F, C, D])), raw = addmm",
)


@torch.library.custom_op(
    "sdm_inference::column_projection_rms",
    mutates_args=(),
    tags=registry.EXACT_STRIDES,
)
def column_projection_rms(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    table: Tensor,
    norm_weight: Tensor,
    eps: float,
    kernel: str,
) -> tuple[Tensor, Tensor]:
    return registry.launch(
        registry.kernel(kernel), x, weight, bias, table, norm_weight, eps
    )


@column_projection_rms.register_fake
def _(x, weight, bias, table, norm_weight, eps, kernel):
    c, _, d = table.shape
    return x.new_empty(x.shape), x.new_empty(x.size(0) // c, c, d)


def _supports(x, weight, bias, table, norm_weight, eps) -> str | None:
    if (
        table.dim() != 3
        or table.size(0) < 1
        or x.dim() != 2
        or table.size(2) != x.size(1)
    ):
        return "needs x [M, D] and a [C, P + F, D] table"
    c, t, _ = table.shape
    if x.size(0) % c or t - x.size(0) // c != READOUTS:
        return (
            f"needs M = C * F and a table of {READOUTS} + F rows per context"
        )
    if (
        x.size(1) not in WIDTHS
        or not sm120.MIN_M <= x.size(0) <= MAX_M[x.size(1)]
    ):
        return f"needs x of shape [{sm120.MIN_M}<=M<={MAX_M}, D in {WIDTHS}]"
    d = x.size(1)
    if weight.shape != (d, d) or bias.shape != (d,):
        return f"needs a {d}x{d} weight and [{d}] bias"
    if (
        norm_weight.dtype != torch.float32
        or norm_weight.shape != (d,)
        or norm_weight.stride() != (1,)
        or norm_weight.storage_offset()
        or norm_weight.device != x.device
    ):
        return f"needs a contiguous FP32 [{d}] norm weight"
    if eps != sm120.EPS:
        return f"needs eps {sm120.EPS}"
    return sm120.policy(x, weight, bias, table)


def _rmsnorm_launches(x: Node) -> list[tuple[Node, Node, float]]:
    """SDM ``rmsnorm_cast`` launches reading ``x``: (output, weight, eps)."""
    found = []
    for launch in x.users:
        if fx_pass.triton_kernel_name(launch) != "_rmsnorm_cast_kernel":
            continue
        tensors = launch.kwargs["kwargs"]
        constants = kernel_side_table.get_constant_args(
            launch.kwargs["constant_args_idx"]
        )
        outputs = list(launch.users)
        if (
            tensors.get("x_ptr") is not x
            or "weight_ptr" not in tensors
            or "cos_ptr" in tensors
            or constants.get("cos_ptr", 0) is not None
            or len(outputs) != 1
            or outputs[0].args[1] != "out_ptr"
        ):
            continue
        if fx_pass.val(outputs[0]).dtype != fx_pass.val(x).dtype:
            continue
        found.append((outputs[0], tensors["weight_ptr"], constants["eps"]))
    return found


def _table_value(node: object) -> bool:
    """``node`` is a value of the in-graph ``[C, P + F, D]`` row table.

    SDM allocates the table with ``torch.empty`` and updates it in place;
    after functionalization each update (``slice_scatter`` into ``t[:, P:]``,
    or the residual ``t + x``) is a new value. A
    ``[C, P + F, D]`` graph input updated in place (copied back at the end)
    counts too.
    """
    pending, seen = [node], set()
    while pending:
        node = pending.pop()
        if node in seen or not isinstance(node, Node):
            continue
        seen.add(node)
        value = node.meta.get("val")
        if not (
            isinstance(value, torch.Tensor)
            and value.dim() == 3
            and value.is_contiguous()
        ):
            continue
        if fx_pass.is_call(node, aten.empty.memory_format) or (
            node.op == "placeholder"
            and any(
                fx_pass.is_call(u, aten.copy_.default) and u.args[0] is node
                for u in node.users
            )
        ):
            return True
        if fx_pass.is_call(node, aten.slice_scatter.default):
            pending.append(node.args[0])
        elif fx_pass.is_call(node, aten.add.Tensor):
            pending.extend(
                a
                for a in node.args
                if isinstance(a, Node) and fx_pass.val(a).shape == value.shape
            )
    return False


def _table_rows(node: object) -> Node | None:
    """The table whose rows ``P:`` ``node`` reads as ``[C, F, D]``.

    Either the flat graph input of ``as_strided(flat, [C, F, D], [(F + P)D,
    D, 1], P * D)`` or the in-graph table ``t`` of ``t[:, P:]``, P = 4.
    """
    skip = READOUTS
    if fx_pass.is_call(node, aten.slice.Tensor):
        t = node.args[0]
        if (
            tuple(node.args[1:3]) == (1, skip)
            and node.args[3:4] != ()
            and node.args[3] >= fx_pass.val(t).size(1)
            and node.args[4:] in ((), (1,))
            and _table_value(t)
        ):
            return t
        return None
    if (
        not fx_pass.is_call(node, aten.as_strided.default)
        or len(node.args) != 4
        or len(node.args[1]) != 3
    ):
        return None
    (_, f, d), stride, start = node.args[1:]
    if list(stride) != [(f + skip) * d, d, 1] or start != skip * d:
        return None
    return node.args[0]


def _full_table(graph: Graph, flat: Node, like: Node) -> Node:
    if fx_pass.val(flat).dim() == 3:
        return flat
    c, f, d = fx_pass.val(like).shape
    t = f + READOUTS
    table = graph.call_function(
        aten.as_strided.default, (flat, [c, t, d], [t * d, d, 1], 0)
    )
    table.meta["val"] = torch.as_strided(
        fx_pass.val(flat), (c, t, d), (t * d, d, 1), 0
    )
    return table


def _other(add: Node, operand: Node) -> object:
    """The other operand of a binary add that reads ``operand``."""
    a, b = add.args
    return b if a is operand else a if b is operand else None


def _values(args):
    return [fx_pass.val(a) if isinstance(a, Node) else a for a in args]


def _column(
    graph: Graph, addmm: Node, view: Node, add: Node, order: dict
) -> bool:
    """add(view(raw), permute(table[:, 4:])) -> rmsnorm."""
    prior = _other(add, view)
    if not fx_pass.is_call(prior, aten.permute.default, [1, 0, 2]):
        return False
    rows = prior.args[0]
    flat = _table_rows(rows)
    norms = _rmsnorm_launches(add)
    if (
        flat is None
        or len(norms) != 1
        or fx_pass.val(view).shape != fx_pass.val(prior).shape
    ):
        return False
    out, norm_weight, eps = norms[0]
    # raw replaces addmm everywhere, so it goes before addmm's first user.
    first = min(addmm.users, key=order.__getitem__)
    if max(order[flat], order[norm_weight]) > order[first]:
        return False
    bias, x, transposed = addmm.args
    with graph.inserting_before(first):
        table = _full_table(graph, flat, rows)
        args = (x, transposed.args[0], bias, table, norm_weight, eps)
        kernel = registry.select("column_projection_rms", *_values(args))
        if kernel is None:
            if table is not flat:
                graph.erase_node(table)
            return False
        fused = graph.call_function(
            torch.ops.sdm_inference.column_projection_rms.default,
            (*args, kernel.name),
        )
        fused.meta["val"] = (fx_pass.val(addmm), fx_pass.val(out))
        raw, norm = (
            graph.call_function(operator.getitem, (fused, i)) for i in range(2)
        )
    raw.meta["val"], norm.meta["val"] = fx_pass.val(addmm), fx_pass.val(out)
    fx_pass.replace(addmm, raw)
    fx_pass.replace(out, norm)
    return True


def _rule(graph: Graph, addmm: Node) -> bool:
    """Match addmm -> reshape -> residual add -> rmsnorm."""
    if not fx_pass.plain_addmm(addmm) or not fx_pass.is_call(
        addmm.args[2], aten.permute.default, [1, 0]
    ):
        return False
    order = {n: i for i, n in enumerate(graph.nodes)}
    for view in list(addmm.users):
        if fx_pass.is_call(view, aten.reshape.default) or fx_pass.is_call(
            view, aten.view.default
        ):
            for add in list(view.users):
                if (
                    fx_pass.is_call(add, aten.add.Tensor)
                    and len(add.args) == 2
                    and not add.kwargs
                    and _column(graph, addmm, view, add, order)
                ):
                    return True
    return False


fx_pass.add_rule(aten.addmm.default, "column_projection_rms", _rule)


def register(directory: Path) -> None:
    module = sm120.load_module(directory, MODULE)

    def impl(x, weight, bias, table, norm_weight, eps):
        c, _, d = table.shape
        raw, norm = torch.empty_like(x), x.new_empty(x.size(0) // c, c, d)
        module.run(x, weight, bias, table, norm_weight, raw, norm, eps)
        return raw, norm

    registry.register(
        registry.Kernel(
            "column_projection_rms",
            "sm120_column_projection_rms",
            _supports,
            impl,
            ("19ColumnProjectionRms",),
        )
    )
