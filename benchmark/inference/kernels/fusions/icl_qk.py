# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-head RMS of the ICL attention query and key, fused with their copies.

``icl_query_rms_scale(q, head_scale, log_n)``: SDM's Triton
``_rmsnorm_cast_kernel`` on the raw query heads, then the LogScale multiply.
The compiled graph stores the RMS in BF16, then computes
``q * log_n * head_scale`` in FP32 (head_scale read as FP32, Inductor drops
its BF16 cast) with one rounding; the kernel does both in one pass.

``icl_key_rms(kv, heads, key, cache)``: the key RMS read in place from the
packed KV addmm output (SDM's ``rmsnorm_cast`` copies the strided K chunk
first), plus the cache head slices ``k[:, :cache]`` and ``v[:, :cache]``
written contiguous. With ``key=False`` (the last ICL fit layer, where only
the cache reads K and V) the rule also narrows the KV addmm to the weight
rows of those cache heads.
"""

import operator
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
from benchmark.inference.kernels.fx_pass import is_call, only_user, val

aten, prims = torch.ops.aten, torch.ops.prims
MODULE = "icl_qk_sm120"
QUERY, KEY = "icl_query_rms_scale", "icl_key_rms"
HEAD = 64
EPS = 1e-6
# ICL widths: Small/Medium 512, Large 1024.
WIDTHS = (512, 1024)
# M range where cuBLAS sums the narrowed KV addmm in the same order as the
# full one (every M to 8192, every 7th to 65,535); it differs for M <= 512.
GATHER_M = (513, 65_535)


def _rms(x: Tensor) -> Tensor:
    return F.rms_norm(x.float(), (HEAD,), eps=EPS).to(torch.bfloat16)


def _query_reference(q: Tensor, head_scale: Tensor, log_n: float) -> Tensor:
    scaled = _rms(q).float() * log_n * head_scale.float()[:, None]
    return scaled.to(torch.bfloat16)


def _key_reference(
    kv: Tensor, heads: int, key: bool, cache: int
) -> tuple[Tensor, Tensor, Tensor]:
    k, v = kv.unflatten(-1, (2, heads, HEAD)).unbind(1)
    k = _rms(k)
    return (
        k if key else k[:, :0].contiguous(),
        k[:, :cache].contiguous(),
        v[:, :cache].contiguous(),
    )


registry.define_op(
    QUERY,
    _query_reference,
    "bf16(rms(q) * log_n * head_scale) per head of 64, eps 1e-6",
)
registry.define_op(
    KEY,
    _key_reference,
    "rms of K read from packed [M, 2 * heads * 64] KV, plus K/V cache heads",
)


@torch.library.custom_op(
    f"sdm_inference::{QUERY}", mutates_args=(), tags=registry.EXACT_STRIDES
)
def icl_query_rms_scale(
    q: Tensor, head_scale: Tensor, log_n: float, kernel: str
) -> Tensor:
    return registry.launch(registry.kernel(kernel), q, head_scale, log_n)


@icl_query_rms_scale.register_fake
def _(q: Tensor, head_scale: Tensor, log_n: float, kernel: str) -> Tensor:
    return q.new_empty(q.shape)


@torch.library.custom_op(
    f"sdm_inference::{KEY}", mutates_args=(), tags=registry.EXACT_STRIDES
)
def icl_key_rms(
    kv: Tensor, heads: int, key: bool, cache: int, kernel: str
) -> tuple[Tensor, Tensor, Tensor]:
    return registry.launch(registry.kernel(kernel), kv, heads, key, cache)


@icl_key_rms.register_fake
def _(
    kv: Tensor, heads: int, key: bool, cache: int, kernel: str
) -> tuple[Tensor, Tensor, Tensor]:
    m = kv.size(0)
    return (
        kv.new_empty(m, heads if key else 0, HEAD),
        kv.new_empty(m, cache, HEAD),
        kv.new_empty(m, cache, HEAD),
    )


def _other(node: Node, first: object) -> object | None:
    """The other operand of binary ``node`` if one operand is ``first``."""
    a, b = node.args[:2]
    return b if a is first else a if b is first else None


def _query_rule(graph: Graph, norm: Node) -> bool:
    """rmsnorm(q) * full(log n) * view(bf16(head_scale), [1, H, 1])."""
    launch = unweighted_rms_launch(norm)
    if launch is None:
        return False
    q, out, channels, eps = launch
    if channels != HEAD or eps != EPS or val(q).dim() != 3:
        return False
    if len(out.users) != 1:
        return False
    (scaled,) = out.users
    if not is_call(scaled, aten.mul.Tensor) or len(scaled.users) != 1:
        return False
    full = _other(scaled, out)
    if not (
        is_call(full, aten.full.default)
        and full.kwargs.get("dtype") == torch.bfloat16
        and val(full).numel() == 1
        and only_user(full, scaled)
    ):
        return False
    log_n = float(full.args[1])
    if torch.tensor(log_n, dtype=torch.bfloat16).item() != log_n:
        return False
    (product,) = scaled.users
    if not is_call(product, aten.mul.Tensor):
        return False
    shaped = _other(product, scaled)
    heads = val(q).shape[1]
    if not (
        is_view(shaped)
        and only_user(shaped, product)
        and tuple(val(shaped).shape) == (1, heads, 1)
    ):
        return False
    cast = shaped.args[0]
    if not (
        is_call(cast, prims.convert_element_type.default, torch.bfloat16)
        and only_user(cast, shaped)
        and val(cast.args[0]).dtype == torch.float32
    ):
        return False
    head_scale = cast.args[0]
    if heads * HEAD not in WIDTHS or val(product).shape != val(q).shape:
        return False
    kernel = registry.select(QUERY, val(q), val(head_scale), log_n)
    if kernel is None:
        return False
    with graph.inserting_before(product):
        fused = graph.call_function(
            torch.ops.sdm_inference.icl_query_rms_scale.default,
            (q, head_scale, log_n, kernel.name),
        )
        fused.meta["val"] = val(q).new_empty(val(q).shape)
    fx_pass.replace(product, fused)
    return True


def _cache_copy(heads: Node) -> tuple[Node, Node, int] | None:
    """The user ``clone(heads[:, :c])`` of ``heads``: (slice, clone, c)."""
    for user in heads.users:
        if not (
            is_call(user, aten.slice.Tensor, 1, 0)
            and len(user.args) == 4
            and isinstance(user.args[3], int)
            and 0 < user.args[3] <= val(heads).shape[1]
            and len(user.users) == 1
        ):
            continue
        (copy,) = user.users
        if is_call(copy, aten.clone.default) and val(copy).is_contiguous():
            return user, copy, user.args[3]
    return None


def _through_clone(node: Node) -> Node:
    """The full ``clone`` of ``node`` if any (SDM caches V contiguous)."""
    clones = [
        u
        for u in node.users
        if is_call(u, aten.clone.default) and val(u).shape == val(node).shape
    ]
    return clones[0] if len(clones) == 1 else node


def _gather_rows(graph: Graph, weight: Node, width: int, rows: int) -> Node:
    """``cat(weight[:rows], weight[width:width + rows])`` before any cast."""
    cast = None
    if is_call(weight, prims.convert_element_type.default):
        cast, weight = weight.args[1], weight.args[0]
    parts = []
    for start in (0, width):
        part = graph.call_function(
            aten.slice.Tensor, (weight, 0, start, start + rows)
        )
        part.meta["val"] = val(weight)[start : start + rows]
        parts.append(part)
    out = graph.call_function(aten.cat.default, (parts, 0))
    out.meta["val"] = torch.cat([val(p) for p in parts])
    if cast is None:
        return out
    converted = graph.call_function(
        prims.convert_element_type.default, (out, cast)
    )
    converted.meta["val"] = val(out).to(cast)
    return converted


def _key_rule(graph: Graph, norm: Node) -> bool:
    """Addmm -> split -> K view -> clone -> rmsnorm; cache head copies."""
    launch = unweighted_rms_launch(norm)
    if launch is None:
        return False
    clone, out, channels, eps = launch
    if channels != HEAD or eps != EPS:
        return False
    if not (is_call(clone, aten.clone.default) and only_user(clone, norm)):
        return False
    shaped = clone.args[0]
    if not (is_view(shaped) and only_user(shaped, clone)):
        return False
    raw_k = shaped.args[0]
    if not (
        is_call(raw_k, operator.getitem)
        and raw_k.args[1] == 0
        and only_user(raw_k, shaped)
    ):
        return False
    split = raw_k.args[0]
    width = val(raw_k).shape[-1]
    heads = width // HEAD
    if not (
        is_call(split, aten.split.Tensor)
        and split.args[1] == width
        and split.args[2:] in ((-1,), (1,))
        and width in WIDTHS
        and tuple(val(shaped).shape[1:]) == (heads, HEAD)
    ):
        return False
    halves = {u.args[1]: u for u in split.users}
    kv = split.args[0]
    if (
        len(split.users) != 2
        or set(halves) != {0, 1}
        or val(kv).dim() != 2
        or val(kv).shape[1] != 2 * width
    ):
        return False
    k_copy = _cache_copy(out)
    v_views = [
        u
        for u in halves[1].users
        if is_view(u) and tuple(val(u).shape[1:]) == (heads, HEAD)
    ]
    v_copy = next(
        (
            c
            for c in (_cache_copy(_through_clone(v)) for v in v_views)
            if c is not None
        ),
        None,
    )
    if k_copy is None or v_copy is None or k_copy[2] != v_copy[2]:
        k_copy = v_copy = None
    cache = 0 if k_copy is None else k_copy[2]
    key = k_copy is None or list(out.users) != [k_copy[0]]
    if not key and cache == 0:
        return False
    gather = (
        not key
        and fx_pass.plain_addmm(kv)
        and only_user(kv, split)
        and list(halves[1].users) == v_views[:1]
        and list(_through_clone(v_views[0]).users) == [v_copy[0]]
        and len(v_views[0].users) == 1
        and is_call(kv.args[2], aten.permute.default, [1, 0])
        and GATHER_M[0] <= val(kv).shape[0] <= GATHER_M[1]
    )
    order = {n: i for i, n in enumerate(graph.nodes)}
    first = min(
        (n for n in (out, v_copy[1] if v_copy else None) if n is not None),
        key=order.__getitem__,
    )
    heads_in = cache if gather else heads
    m = val(kv).shape[0]
    like = val(kv).new_empty(m, 2 * heads_in * HEAD)
    kernel = registry.select(KEY, like, heads_in, key, cache)
    if kernel is None:
        return False
    source = kv
    if gather:
        bias, x, transposed = kv.args
        rows = cache * HEAD
        with graph.inserting_before(first):
            weight = _gather_rows(graph, transposed.args[0], width, rows)
            bias = _gather_rows(graph, bias, width, rows)
            permuted = graph.call_function(
                aten.permute.default, (weight, [1, 0])
            )
            permuted.meta["val"] = val(weight).permute(1, 0)
            source = graph.call_function(
                aten.addmm.default, (bias, x, permuted)
            )
            source.meta["val"] = like
    with graph.inserting_before(first):
        fused = graph.call_function(
            torch.ops.sdm_inference.icl_key_rms.default,
            (source, heads_in, key, cache, kernel.name),
        )
        fused.meta["val"] = (
            val(source).new_empty(m, heads_in if key else 0, HEAD),
            val(source).new_empty(m, cache, HEAD),
            val(source).new_empty(m, cache, HEAD),
        )
        parts = []
        for i in range(3):
            parts.append(graph.call_function(operator.getitem, (fused, i)))
            parts[-1].meta["val"] = fused.meta["val"][i]
    if key:
        fx_pass.replace(out, parts[0])
    if cache:
        fx_pass.replace(k_copy[1], parts[1])
        fx_pass.replace(v_copy[1], parts[2])
    return True


fx_pass.add_rule(fx_pass.TRITON_KERNEL, QUERY, _query_rule)
fx_pass.add_rule(fx_pass.TRITON_KERNEL, KEY, _key_rule)


def _device(t: Tensor) -> str | None:
    if t.device.type != "cuda":
        return "needs CUDA tensors"
    if torch.cuda.get_device_capability(t.device) != (12, 0):
        return "needs SM 12.0"
    if t.dtype != torch.bfloat16:
        return "needs bfloat16"
    return None


def _rows(t: Tensor, width: int, what: str) -> str | None:
    if t.stride(-1) != 1 or t.stride(0) < width:
        return f"needs {what} rows of {width} with unit stride"
    if t.stride(0) % 8 or t.storage_offset() % 8:
        return f"needs 16-byte aligned {what} rows"
    return None


def _query_supports(q: Tensor, head_scale: Tensor, log_n: float) -> str | None:
    if q.dim() != 3 or q.shape[2] != HEAD or q.stride(1) != HEAD:
        return "needs q [M, H, 64] with head stride 64"
    if q.numel() == 0:
        return "needs a nonempty q"
    if head_scale.dtype != torch.float32 or head_scale.shape != q.shape[1:2]:
        return "needs FP32 head_scale [H]"
    if not head_scale.is_contiguous() or head_scale.device != q.device:
        return "needs contiguous head_scale on q's device"
    return _device(q) or _rows(q, q.shape[1] * HEAD, "q")


def _key_supports(kv: Tensor, heads: int, key: bool, cache: int) -> str | None:
    if kv.dim() != 2 or kv.shape[1] != 2 * heads * HEAD or kv.numel() == 0:
        return "needs nonempty kv [M, 2 * heads * 64]"
    if not 0 <= cache <= heads or not (key or cache):
        return "needs 0 <= cache <= heads and some output"
    return _device(kv) or _rows(kv, kv.shape[1], "kv")


def register(directory: Path) -> None:
    module = sm120.load_module(directory, MODULE)

    def query(q: Tensor, head_scale: Tensor, log_n: float) -> Tensor:
        return module.query(q, head_scale, log_n)

    def key(
        kv: Tensor, heads: int, key: bool, cache: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(module.key(kv, heads, key, cache))

    registry.register(
        registry.Kernel(
            op=QUERY,
            name="sm120_icl_query_rms_scale",
            supports=_query_supports,
            impl=query,
            cuda_symbols=("icl_qk::query_kernel",),
        )
    )
    registry.register(
        registry.Kernel(
            op=KEY,
            name="sm120_icl_key_rms",
            supports=_key_supports,
            impl=key,
            cuda_symbols=("icl_qk::key_kernel",),
        )
    )
