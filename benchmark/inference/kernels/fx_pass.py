# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inductor post-grad pass that routes matched regions to selected kernels.

Matches the pre-fusion ATen form of each named op. A region is rewritten only
if :func:`registry.select` admits its compile-time tensor metadata; otherwise
the graph is left untouched for Inductor.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torch._higher_order_ops.triton_kernel_wrap import (
    kernel_side_table,
    triton_kernel_wrapper_functional,
)
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch.fx import Graph, Node

from benchmark.inference.kernels import registry

aten = torch.ops.aten


def is_call(node: object, target: object, *args: object) -> bool:
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == target
        and all(a is None or a == b for a, b in zip(args, node.args[1:]))
    )


def plain_addmm(node: object) -> bool:
    """``addmm(bias, x, w)`` with the default alpha and beta."""
    return (
        is_call(node, aten.addmm.default)
        and len(node.args) == 3
        and not node.kwargs
    )


def only_user(node: Node, user: Node) -> bool:
    return list(node.users) == [user]


def val(node: Node) -> torch.Tensor:
    return node.meta["val"]


TRITON_KERNEL = triton_kernel_wrapper_functional


def triton_kernel_name(node: Node) -> str | None:
    """Name of the user-defined Triton kernel ``node`` launches, if any."""
    if not is_call(node, TRITON_KERNEL):
        return None
    kernel = kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
    while not hasattr(kernel, "__name__") and hasattr(kernel, "fn"):
        kernel = kernel.fn
    return getattr(kernel, "__name__", None)


Rule = Callable[[Graph, Node], bool]
RULES: list[tuple[object, str, Rule]] = []


def add_rule(target: object, op: str, rule: Rule) -> None:
    """Try ``rule`` at nodes calling ``target``, after earlier rules."""
    RULES.append((target, op, rule))


_shadow: list[str] = []


@torch.library.custom_op(
    "sdm_inference::shadow_compare",
    mutates_args=(),
    tags=registry.EXACT_STRIDES,
)
def shadow_compare(native: Tensor, kernel: Tensor, op: str) -> Tensor:
    registry.record_shadow(op, native, kernel)
    return _like(native).copy_(native)


@shadow_compare.register_fake
def _(native: Tensor, kernel: Tensor, op: str) -> Tensor:
    return _like(native)


def _like(t: Tensor) -> Tensor:
    return t.new_empty_strided(t.shape, t.stride())


def replace(old: Node, new: Node) -> None:
    """Route the users of native ``old`` to the kernel's ``new``.

    In shadow mode ``old`` stays on the data path and every later user reads
    ``shadow_compare(old, new)``, which records whether the two are bitwise
    equal, so each kernel is checked on the exact inputs native sees.
    """
    if not _shadow:
        old.replace_all_uses_with(new)
        return
    graph = old.graph
    order = {n: i for i, n in enumerate(graph.nodes)}
    last = max(old, new, key=order.__getitem__)
    with graph.inserting_after(last):
        compare = graph.call_function(
            torch.ops.sdm_inference.shadow_compare.default,
            (old, new, _shadow[-1]),
        )
    compare.meta["val"] = val(old)
    old.replace_all_uses_with(
        compare,
        delete_user_cb=lambda u: (
            u is not compare and order.get(u, len(order)) > order[last]
        ),
    )


class KernelPass(CustomGraphPass):
    def __init__(
        self, ops: frozenset[str] | None = None, shadow: bool = False
    ) -> None:
        self.rules: dict = {}
        self.shadow = shadow
        for target, op, rule in RULES:
            if ops is None or op in ops:
                self.rules.setdefault(target, []).append((op, rule))

    def __call__(self, graph: Graph) -> None:
        for node in list(graph.nodes):
            if node.op == "call_function":
                for op, rule in self.rules.get(node.target, ()):
                    if self.shadow:
                        _shadow.append(op)
                    try:
                        if rule(graph, node):
                            break
                    finally:
                        if self.shadow:
                            _shadow.pop()
        graph.eliminate_dead_code()

    def uuid(self) -> None:
        return None  # never cache graphs compiled with this pass


def compile_options(
    ops: frozenset[str] | None = None, shadow: bool = False
) -> dict:
    """Inductor options adding the pass; ``ops`` limits it to those ops.

    ``shadow`` keeps native on the data path and compares each kernel output
    with the native value it replaces (see :func:`replace`).
    """
    return {"post_grad_custom_post_pass": KernelPass(ops, shadow)}
