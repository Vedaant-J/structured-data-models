# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-attention over the context rows (the ICL fit site).

``icl_attention(q, k, v, scale)`` is ``F.scaled_dot_product_attention`` on
``[B, S, H, 64]`` memory (the ``[B, H, S, D]`` graph tensors transposed),
returned as ``[B, S, H, 64]``. The pass replaces PyTorch's flash node with
this op followed by the permute that restores the flash output strides.

None of these kernels is bitwise equal to PyTorch. ``$SDM_ICL_ATTENTION`` (or
the ``attention`` of :func:`registry.enabled`) picks one:

- ``sm120`` (default): BF16 SM120 kernel, qualified by the tolerance gate;
- ``fa2``: FlashAttention 2, if installed;
- ``int8[+ends]``: opt-in lossy INT8 QK / FP8 PV kernel with K smoothing
  (``ends``: keep the first and last ICL fit call in PyTorch), gated
  ``opt-in``;
- ``sm120+uncapped``: diagnostic, the BF16 kernel above ``MAX_S``;
- ``off``: PyTorch. ``kernels.enable(model, exact=True)`` also keeps it.
"""

import functools
import operator
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fx import Graph, Node

from benchmark.inference.kernels import fx_pass, registry
from benchmark.inference.kernels.fx_pass import is_call, val
from benchmark.inference.kernels.registry import Kernel
from benchmark.inference.kernels.sm120 import load_module

aten = torch.ops.aten
MODULE = "icl_attention_sm120"
INT8_MODULE = "icl_attention_int8_sm120"
ENV = registry.ATTENTION_ENV
HEAD = 64
MIN_S = 1024
# Largest context the BF16 kernel's error and determinism were checked at;
# Medium at 100k misses the tolerance gate.
MAX_S = 65_535
FA2_SYMBOL = "void flash::flash_fwd_kernel<"
INT8_SYMBOLS = (
    "icl_attention_int8::quant_kernel",
    "icl_attention_int8::attend_kernel",
)
BSHD = [0, 2, 1, 3]


def _reference(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
    ).transpose(1, 2)


registry.define_op(
    "icl_attention",
    _reference,
    "softmax(q k^T * scale) v on [B, S, H, 64], Sq == Skv, no mask",
)


@torch.library.custom_op(
    "sdm_inference::icl_attention",
    mutates_args=(),
    tags=registry.EXACT_STRIDES,
)
def icl_attention(
    q: Tensor, k: Tensor, v: Tensor, scale: float, kernel: str
) -> Tensor:
    return registry.launch(registry.kernel(kernel), q, k, v, scale)


@icl_attention.register_fake
def _(q, k, v, scale, kernel):
    return q.new_empty(q.shape)


def _admits(
    q: Tensor, k: Tensor, v: Tensor, scale: float, dtypes=(torch.bfloat16,)
) -> str | None:
    if any(t.device.type != "cuda" for t in (q, k, v)):
        return "needs CUDA tensors"
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        return "needs SM 12.0"
    if q.dtype not in dtypes or any(t.dtype != q.dtype for t in (k, v)):
        return f"needs q, k, v all one of {[str(d) for d in dtypes]}"
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return "no backward"
    if q.dim() != 4 or q.shape != k.shape or q.shape != v.shape:
        return "needs q, k, v [B, S, H, 64] of one shape (self-attention)"
    s, d = q.shape[1], q.shape[3]
    if d != HEAD:
        return f"needs head dim {HEAD}"
    if s < MIN_S:
        return f"needs S >= {MIN_S} (fit); got {s}"
    for t in (q, k, v):
        if t.stride(3) != 1 or any(x % 8 for x in t.stride()[:3]):
            return "needs last-dim stride 1 and 16-byte aligned rows"
        if t.storage_offset() % 8:
            return "needs a 16-byte aligned storage offset"
    return None


def _bf16_limits(q: Tensor) -> str | None:
    if q.size(1) > MAX_S:
        return f"needs S <= {MAX_S} (checked); got {q.size(1)}"
    return None


def _int8_limits(q: Tensor, k: Tensor, v: Tensor) -> str | None:
    b, s, h, _ = q.shape
    if b != 1:
        return f"needs batch 1 (the ICL fit site); got {b}"
    rows = max(t.stride(1) for t in (q, k, v))
    if s * rows >= 2**31 or b * h * s * HEAD >= 2**31:
        return "needs 32-bit element offsets"
    return None


def _setting() -> tuple[str, frozenset[str]]:
    return registry.parse_attention(registry.attention())


def _policy(want: str):
    def supports(q: Tensor, k: Tensor, v: Tensor, scale: float):
        try:
            backend, chosen = _setting()
        except ValueError as error:
            return str(error)
        if backend != want or chosen - {"ends", "uncapped"}:
            return f"{ENV}={registry.attention()}"
        if want == "int8":
            return _admits(
                q, k, v, scale, (torch.bfloat16, torch.float16)
            ) or _int8_limits(q, k, v)
        return _admits(q, k, v, scale) or (
            None if "uncapped" in chosen else _bf16_limits(q)
        )

    return supports


@functools.cache
def _flash_attn_func():
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None
    return flash_attn_func


def _fa2_supports(q: Tensor, k: Tensor, v: Tensor, scale: float):
    if registry.attention() == "fa2" and _flash_attn_func() is None:
        return "flash_attn is not importable"
    return _policy("fa2")(q, k, v, scale)


def _fa2_impl(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    return _flash_attn_func()(q, k, v, softmax_scale=scale)


def _flash_node(node: Node) -> tuple[tuple[Node, ...], Node] | None:
    """(q, k, v) and the sole getitem 0 of a plain flash SDPA node."""
    if not is_call(node, aten._scaled_dot_product_flash_attention.default):
        return None
    if len(node.args) != 3 or not set(node.kwargs) <= {
        "dropout_p",
        "is_causal",
        "return_debug_mask",
        "scale",
    }:
        return None
    kwargs = node.kwargs
    if (
        kwargs.get("dropout_p", 0.0) != 0.0
        or kwargs.get("is_causal", False)
        or kwargs.get("return_debug_mask", False)
        or kwargs.get("scale") is None
    ):
        return None
    if not (
        len(node.users) == 1
        and all(is_call(u, operator.getitem, 0) for u in node.users)
    ):
        return None
    return node.args, next(iter(node.users))


def _fit_call(node: Node) -> bool:
    """A batch-1 self-attention flash call: an ICL fit call."""
    found = _flash_node(node)
    if found is None:
        return False
    q, k, _ = (val(t) for t in found[0])
    return q.shape == k.shape and q.size(0) == 1 and q.size(3) == HEAD


def _rule(graph: Graph, node: Node) -> bool:
    found = _flash_node(node)
    if found is None:
        return False
    try:
        _, options = _setting()
    except ValueError:
        options = frozenset()
    if "ends" in options and _fit_call(node):
        calls = [n for n in graph.nodes if _fit_call(n)]
        if node in (calls[0], calls[-1]):
            return False
    (q, k, v), out = found
    scale = float(node.kwargs["scale"])
    kernel = registry.select(
        "icl_attention", *(val(t).permute(BSHD) for t in (q, k, v)), scale
    )
    if kernel is None:
        return False
    with graph.inserting_before(out):
        views = []
        for t in (q, k, v):
            views.append(graph.call_function(aten.permute.default, (t, BSHD)))
            views[-1].meta["val"] = val(t).permute(BSHD)
        fused = graph.call_function(
            torch.ops.sdm_inference.icl_attention.default,
            (*views, scale, kernel.name),
        )
        fused.meta["val"] = val(views[0]).new_empty(val(views[0]).shape)
        result = graph.call_function(aten.permute.default, (fused, BSHD))
        result.meta["val"] = val(fused).permute(BSHD)
    fx_pass.replace(out, result)
    return True


fx_pass.add_rule(
    aten._scaled_dot_product_flash_attention.default, "icl_attention", _rule
)


def register(directory: Path) -> None:
    module = load_module(directory, MODULE)

    def impl(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
        return module.run(q, k, v, q.new_empty(q.shape), scale)

    registry.register(
        Kernel(
            op="icl_attention",
            name="sm120_icl_attention_d64",
            supports=_policy("sm120"),
            impl=impl,
            cuda_symbols=("icl_attention_kernel",),
            bitwise=lambda *args: False,
        )
    )
    registry.register(
        Kernel(
            op="icl_attention",
            name="fa2_icl_attention_d64",
            supports=_fa2_supports,
            impl=_fa2_impl,
            cuda_symbols=(FA2_SYMBOL,),
            bitwise=lambda *args: False,
        )
    )
    int8 = load_module(directory, INT8_MODULE)

    def int8_impl(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
        out = q.new_empty(q.shape)
        int8.run(*(t.permute(BSHD) for t in (q, k, v, out)), scale)
        return out

    registry.register(
        Kernel(
            op="icl_attention",
            name="sm120_int8_icl",
            supports=_policy("int8"),
            impl=int8_impl,
            cuda_symbols=INT8_SYMBOLS,
            bitwise=lambda *args: False,
            exact=False,
        )
    )
