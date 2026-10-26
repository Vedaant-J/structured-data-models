# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row-block Q/K transform: split-half RoPE then per-head RMS, one launch.

``rope_rms(q, k, inv_freq) = (rms(rope(q)), rms(rope(k)))`` for ``q, k
[C, S, 4, head]`` (head 32 or 64), FP32 ``inv_freq [head / 2]``, positions
``0..S-1`` along dim 1, eps 1e-6, no weight.

Compiled SDM runs Q and K each as Inductor's rope pointwise feeding SDM's
Triton ``_rmsnorm_cast_kernel``. The pass pairs two such regions sharing
``inv_freq`` and shape (a layer's Q and K) into one call. In the last row
layer the queries are only the 4 readout tokens, so Q and K differ in length
there; ``rope_rms_uneven`` covers that case.
"""

import operator
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Size, Tensor
from torch.fx import Graph, Node

from benchmark.inference.kernels import fx_pass, registry
from benchmark.inference.kernels.fusions.kv_rms import unweighted_rms_launch
from benchmark.inference.kernels.fx_pass import is_call, val
from benchmark.inference.kernels.registry import Kernel
from benchmark.inference.kernels.sm120 import load_module

aten, prims = torch.ops.aten, torch.ops.prims
MODULE = "rope_rms_sm120"
EPS = 1e-6
HEADS = 4
WIDTHS = (32, 64)
MAX_S = 1024  # every position below it is checked against compiled native
# Head width -> largest C * S checked bitwise against compiled native.
MAX_TOKENS = {32: 20_234_480, 64: 8_388_737}


def _reference(
    q: Tensor, k: Tensor, inv_freq: Tensor
) -> tuple[Tensor, Tensor]:
    def one(x: Tensor) -> Tensor:
        angle = (
            torch.arange(x.size(1), device=x.device, dtype=torch.float32)[
                :, None
            ]
            * inv_freq
        )
        cos, sin = angle.cos()[:, None], angle.sin()[:, None]
        x1, x2 = x.float().chunk(2, -1)
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).to(
            x.dtype
        )
        return F.rms_norm(rotated.float(), (x.size(-1),), eps=EPS).to(x.dtype)

    return one(q), one(k)


registry.define_op(
    "rope_rms",
    _reference,
    "(rms(rope(q)), rms(rope(k))): split-half RoPE, per-head RMS, eps 1e-6",
)


@torch.library.custom_op(
    "sdm_inference::rope_rms", mutates_args=(), tags=registry.EXACT_STRIDES
)
def rope_rms(
    q: Tensor, k: Tensor, inv_freq: Tensor, kernel: str
) -> tuple[Tensor, Tensor]:
    return registry.launch(registry.kernel(kernel), q, k, inv_freq)


@rope_rms.register_fake
def _(q, k, inv_freq, kernel):
    return q.new_empty(q.shape), q.new_empty(q.shape)


registry.define_op(
    "rope_rms_uneven",
    _reference,
    "rope_rms with q [C, Sq, 4, head] and k [C, Sk, 4, head]",
)


@torch.library.custom_op(
    "sdm_inference::rope_rms_uneven",
    mutates_args=(),
    tags=registry.EXACT_STRIDES,
)
def rope_rms_uneven(
    q: Tensor, k: Tensor, inv_freq: Tensor, kernel: str
) -> tuple[Tensor, Tensor]:
    return registry.launch(registry.kernel(kernel), q, k, inv_freq)


@rope_rms_uneven.register_fake
def _(q, k, inv_freq, kernel):
    return q.new_empty(q.shape), k.new_empty(k.shape)


def _bf16(node: object, trig: object) -> Node | None:
    """convert(unsqueeze(trig(angle), 1 or -2), bf16) -> angle."""
    if not (
        is_call(node, prims.convert_element_type.default)
        and node.args[1] == torch.bfloat16
    ):
        return None
    unsqueeze = node.args[0]
    if not (
        is_call(unsqueeze, aten.unsqueeze.default)
        and unsqueeze.args[1] in (1, -2)
        and is_call(unsqueeze.args[0], trig)
        and val(unsqueeze.args[0]).dim() == 2
    ):
        return None
    return unsqueeze.args[0].args[0]


def _strip(node: object, target: object, value: int) -> object:
    return (
        node.args[0]
        if is_call(node, target) and node.args[1] == value
        else node
    )


def _positions(node: object, seq: int) -> bool:
    """Node = convert(iota(seq) [* 1] [+ 0], f32)."""
    if not (
        is_call(node, prims.convert_element_type.default)
        and node.args[1] == torch.float32
    ):
        return False
    iota = _strip(_strip(node.args[0], aten.add.Tensor, 0), aten.mul.Tensor, 1)
    return (
        is_call(iota, prims.iota.default)
        and iota.args[0] == seq
        and iota.kwargs.get("start", 0) == 0
        and iota.kwargs.get("step", 1) == 1
    )


def _inv_freq(angle: object, seq: int) -> Node | None:
    """Angle = positions as [S, 1] * inv_freq (or inv_freq as [1, -1])."""
    if not is_call(angle, aten.mul.Tensor):
        return None
    pos, freq = angle.args
    views = (aten.reshape.default, aten.view.default)
    if any(is_call(pos, v) for v in views):
        if list(pos.args[1]) != [-1, 1]:
            return None
    elif not (is_call(pos, aten.unsqueeze.default) and pos.args[1] in (1, -1)):
        return None
    if any(is_call(freq, v) for v in views) and list(freq.args[1]) == [1, -1]:
        freq = freq.args[0]
    if not isinstance(freq, Node) or val(freq).dim() != 1:
        return None
    return freq if _positions(pos.args[0], seq) else None


def _region(norm: Node) -> tuple[Node, Node, Node] | None:
    """rmsnorm(cat([x1 cos - x2 sin, x2 cos + x1 sin], -1)): (x, inv_freq)."""
    launch = unweighted_rms_launch(norm)
    if launch is None:
        return None
    cat, out, channels, eps = launch
    if eps != EPS or not out.users:
        return None
    if not (
        is_call(cat, aten.cat.default)
        and len(cat.args) == 2
        and cat.args[1] in (-1, 3)
    ):
        return None
    if len(cat.args[0]) != 2 or list(cat.users) != [norm]:
        return None
    first, second = cat.args[0]
    if not (
        is_call(first, aten.sub.Tensor) and is_call(second, aten.add.Tensor)
    ):
        return None
    products = (*first.args, *second.args)
    if not all(
        is_call(p, aten.mul.Tensor) and len(p.args) == 2 for p in products
    ):
        return None
    (x1, cos), (x2, sin), (x2b, cos_b), (x1b, sin_b) = (
        p.args for p in products
    )
    if x1 is not x1b or x2 is not x2b or cos is not cos_b or sin is not sin_b:
        return None
    if not (
        is_call(x1, operator.getitem)
        and is_call(x2, operator.getitem)
        and x1.args[0] is x2.args[0]
    ):
        return None
    split = x1.args[0]
    if not (
        is_call(split, aten.split.Tensor)
        and x1.args[1] == 0
        and x2.args[1] == 1
    ):
        return None
    x = split.args[0]
    if not isinstance(x, Node) or val(x).dim() != 4:
        return None
    seq, head = val(x).shape[1], val(x).shape[3]
    if (
        split.args[1] != head // 2
        or split.args[2:] not in ((-1,), (3,))
        or channels != head
    ):
        return None
    angle = _bf16(cos, aten.cos.default)
    if angle is None or _bf16(sin, aten.sin.default) is not angle:
        return None
    inv_freq = _inv_freq(angle, seq)
    return None if inv_freq is None else (x, inv_freq, out)


def _pair_rule(op: str, pairs: Callable[[Size, Size], bool]) -> fx_pass.Rule:
    """Fuse a region with the next one sharing ``inv_freq`` that ``pairs``."""

    def rule(graph: Graph, norm: Node) -> bool:
        first = _region(norm)
        if first is None:
            return False
        order = {n: i for i, n in enumerate(graph.nodes)}
        later = (
            (n, r)
            for n in graph.nodes
            if order[n] > order[norm]
            and fx_pass.triton_kernel_name(n) == "_rmsnorm_cast_kernel"
            for r in [_region(n)]
            if r is not None
            and r[1] is first[1]
            and pairs(val(first[0]).shape, val(r[0]).shape)
        )
        pair = next(later, None)
        if pair is None:
            return False
        second_norm, second = pair
        if min(order[u] for u in first[2].users) < order[second_norm]:
            return False
        (q, inv_freq, q_out), (k, _, k_out) = first, second
        kernel = registry.select(op, val(q), val(k), val(inv_freq))
        if kernel is None:
            return False
        with graph.inserting_before(second_norm):
            fused = graph.call_function(
                getattr(torch.ops.sdm_inference, op).default,
                (q, k, inv_freq, kernel.name),
            )
            fused.meta["val"] = (val(q_out), val(k_out))
            outputs = []
            for i, like in enumerate((q_out, k_out)):
                outputs.append(
                    graph.call_function(operator.getitem, (fused, i))
                )
                outputs[-1].meta["val"] = val(like)
        for old, new in zip((q_out, k_out), outputs):
            fx_pass.replace(old, new)
        return True

    return rule


def _uneven(q: Size, k: Size) -> bool:
    return q[1] != k[1] and (q[0], *q[2:]) == (k[0], *k[2:])


fx_pass.add_rule(
    fx_pass.TRITON_KERNEL, "rope_rms", _pair_rule("rope_rms", operator.eq)
)
fx_pass.add_rule(
    fx_pass.TRITON_KERNEL,
    "rope_rms_uneven",
    _pair_rule("rope_rms_uneven", _uneven),
)


def supports(q: Tensor, k: Tensor, inv_freq: Tensor) -> str | None:
    if (
        any(t.device.type != "cuda" for t in (q, k, inv_freq))
        or len({q.device, k.device, inv_freq.device}) != 1
    ):
        return "needs CUDA tensors on one device"
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        return "needs SM 12.0"
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or inv_freq.dtype != torch.float32
    ):
        return "needs bfloat16 q, k and float32 inv_freq"
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, inv_freq)
    ):
        return "no backward"
    if q.dim() != 4 or q.shape != k.shape:
        return "needs q, k [C, S, 4, head] of one shape"
    c, s, h, head = q.shape
    if h != HEADS or head not in WIDTHS:
        return f"needs {HEADS} heads of width in {WIDTHS}"
    if not 1 <= s <= MAX_S:
        return f"needs 1 <= S <= {MAX_S} (qualified positions)"
    if not 1 <= c * s <= MAX_TOKENS[head]:
        return f"needs 1 <= C * S <= {MAX_TOKENS[head]}"
    for t in (q, k):
        ld = t.stride(1)
        if t.stride() != (s * ld, ld, head, 1) or ld < HEADS * head:
            return "needs q, k strides (S * ld, ld, head, 1), ld >= 4 * head"
        if ld % 4 or t.storage_offset() % 4:
            return "needs 8-byte aligned rows"
    if inv_freq.shape != (head // 2,) or inv_freq.stride() != (1,):
        return "needs contiguous inv_freq [head / 2]"
    return None


def _uneven_supports(q: Tensor, k: Tensor, inv_freq: Tensor) -> str | None:
    if q.dim() != 4 or k.dim() != 4 or q.shape[1] == k.shape[1]:
        return "needs q, k [C, S, 4, head] of different S (else rope_rms)"
    if (q.shape[0], *q.shape[2:]) != (k.shape[0], *k.shape[2:]):
        return "needs q, k of one C, heads and head width"
    longer = q if q.shape[1] > k.shape[1] else k
    return (
        supports(longer, longer, inv_freq)
        or supports(q, q, inv_freq)
        or supports(k, k, inv_freq)
    )


def register(directory: Path) -> None:
    module = load_module(directory, MODULE)

    def impl(q: Tensor, k: Tensor, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
        return tuple(module.run(q, k, inv_freq))

    def uneven_impl(
        q: Tensor, k: Tensor, inv_freq: Tensor
    ) -> tuple[Tensor, Tensor]:
        return tuple(module.run_uneven(q, k, inv_freq))

    registry.register(
        Kernel(
            op="rope_rms",
            name="sm120_rope_rms",
            supports=supports,
            impl=impl,
            cuda_symbols=("rope_rms_kernel",),
        )
    )
    registry.register(
        Kernel(
            op="rope_rms_uneven",
            name="sm120_rope_rms_uneven",
            supports=_uneven_supports,
            impl=uneven_impl,
            cuda_symbols=("rope_rms_uneven_kernel",),
        )
    )
