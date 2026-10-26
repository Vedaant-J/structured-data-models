# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named inference operations with optional kernels and native fallbacks.

An :class:`Op` names an operation and its pure-PyTorch reference semantics.
A :class:`Kernel` implements one op for the inputs its ``supports`` admits.
Kernels are only selectable inside :func:`enabled`; everywhere else, and
whenever no kernel admits the inputs, the region stays native.
"""

import contextlib
import os
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

Output = Tensor | tuple[Tensor, ...] | None
# Ops read tables and views of the in-graph row table; Inductor must hand
# them the strides their fake implementations declare.
EXACT_STRIDES = (torch.Tag.needs_exact_strides,)


@dataclass(frozen=True)
class Op:
    """A named operation.

    Args:
        name: Unique op name.
        reference: Pure-PyTorch implementation defining the semantics.
        doc: One-line description.
    """

    name: str
    reference: Callable[..., Output]
    doc: str


@dataclass(frozen=True)
class Kernel:
    """An implementation of an :class:`Op`.

    Args:
        op: Name of the implemented op.
        name: Unique kernel name.
        supports: Takes the op arguments and returns ``None`` if the kernel
            admits them, else a short reason. Inspects only metadata (shape,
            dtype, stride, device, compute capability); never launches work
            or synchronizes.
        impl: Takes the op arguments and returns new tensors. It mutates
            only the arguments its op declares in ``mutates_args``.
        cuda_symbols: Substrings of the CUDA kernel names ``impl`` launches,
            used to confirm in a profile that it ran.
        bitwise: Takes the op arguments and returns whether they lie where
            the kernel was checked bitwise equal to native; above that,
            ``supports`` may still admit them within a measured tolerance.
        exact: ``False`` for kernels that are lossy by design (opt-in 8-bit
            attention); the benchmark then records the distance instead of
            gating on it.
    """

    op: str
    name: str
    supports: Callable[..., str | None]
    impl: Callable[..., Output]
    cuda_symbols: tuple[str, ...]
    bitwise: Callable[..., bool] = lambda *args, **kwargs: True
    exact: bool = True


OPS: dict[str, Op] = {}
KERNELS: dict[str, list[Kernel]] = {}

ATTENTION_ENV = "SDM_ICL_ATTENTION"
ATTENTION_BACKENDS = ("sm120", "fa2", "int8", "off")
ATTENTION_OPTIONS = ("ends",)
# Diagnostic only: lifts the BF16 kernel's checked row cap.
BF16_OPTIONS = ("uncapped",)

_enabled = False
_attention: str | None = None
_decisions: list[dict] = []
_launches: Counter[str] = Counter()
_shadow: dict[str, dict] = {}


def define_op(name: str, reference: Callable[..., Output], doc: str) -> Op:
    if name in OPS:
        raise ValueError(f"op {name!r} is already defined")
    OPS[name] = op = Op(name=name, reference=reference, doc=doc)
    return op


def register(kernel: Kernel) -> Kernel:
    """Add ``kernel``; earlier registrations take precedence."""
    if kernel.op not in OPS:
        raise ValueError(
            f"kernel {kernel.name!r} implements unknown op {kernel.op!r}"
        )
    if any(k.name == kernel.name for k in registered()):
        raise ValueError(f"kernel {kernel.name!r} is already registered")
    KERNELS.setdefault(kernel.op, []).append(kernel)
    return kernel


def registered() -> list[Kernel]:
    return [kernel for kernels in KERNELS.values() for kernel in kernels]


def parse_attention(setting: str) -> tuple[str, frozenset[str]]:
    """Split ``<backend>[+option...]`` into the backend and its options."""
    backend, *options = setting.split("+")
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(
            f"ICL attention must be one of {ATTENTION_BACKENDS}, "
            f"got {backend!r}"
        )
    allowed = {"int8": ATTENTION_OPTIONS, "sm120": BF16_OPTIONS}.get(
        backend, ()
    )
    if options and not allowed:
        raise ValueError(f"ICL attention {backend!r} takes no options")
    if set(options) - set(allowed) or len(set(options)) != len(options):
        raise ValueError(
            f"ICL attention {backend!r} options must be distinct and from "
            f"{allowed}, got {options}"
        )
    return backend, frozenset(options)


def attention() -> str:
    """The ICL attention setting of :func:`enabled`, else the environment."""
    if _attention is not None:
        return _attention
    return os.environ.get(ATTENTION_ENV, "sm120")


@contextlib.contextmanager
def enabled(attention: str | None = None) -> Iterator[None]:
    """Make registered kernels selectable within this context.

    ``attention`` overrides ``$SDM_ICL_ATTENTION`` (``<backend>[+option...]``,
    see :func:`parse_attention`) for the ICL attention kernels.
    """
    global _enabled, _attention
    if attention is not None:
        parse_attention(attention)
    previous = _enabled, _attention
    _enabled, _attention = True, attention
    try:
        yield
    finally:
        _enabled, _attention = previous


def _describe(value: Any) -> object:
    if isinstance(value, Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "device": str(value.device),
        }
    return repr(value)


def select(op: str, *args: Any, **kwargs: Any) -> Kernel | None:
    """The first registered kernel admitting the arguments, else ``None``.

    Inside :func:`enabled`, every decision is recorded in :func:`census`.
    """
    if not _enabled:
        return None
    chosen, rejected = None, {}
    for kernel in KERNELS.get(op, ()):
        reason = kernel.supports(*args, **kwargs)
        if reason is None:
            chosen = kernel
            break
        rejected[kernel.name] = reason
    _decisions.append(
        {
            "op": op,
            "kernel": "native" if chosen is None else chosen.name,
            "bitwise": chosen is None or chosen.bitwise(*args, **kwargs),
            "exact": chosen is None or chosen.exact,
            "rejected": rejected,
            "args": [_describe(arg) for arg in args],
            "kwargs": {key: _describe(value) for key, value in kwargs.items()},
        }
    )
    return chosen


def kernel(name: str) -> Kernel:
    return next(k for k in registered() if k.name == name)


def launch(kernel: Kernel, *args: Any, **kwargs: Any) -> Output:
    _launches[kernel.name] += 1
    return kernel.impl(*args, **kwargs)


def record_shadow(op: str, native: Tensor, kernel: Tensor) -> None:
    """Shadow mode: whether ``kernel`` reproduced ``native`` bit for bit."""
    entry = _shadow.setdefault(
        op, {"outputs": 0, "bitwise": 0, "max_abs_diff": 0.0}
    )
    entry["outputs"] += 1
    if native.shape != kernel.shape or native.dtype != kernel.dtype:
        entry["max_abs_diff"] = float("inf")
        return
    same = torch.equal(
        native.contiguous().view(torch.uint8).flatten(),
        kernel.contiguous().view(torch.uint8).flatten(),
    )
    entry["bitwise"] += same
    if not same:
        diff = (native.float() - kernel.float()).abs().nan_to_num(1e30)
        entry["max_abs_diff"] = max(entry["max_abs_diff"], diff.max().item())


def census() -> dict[str, Any]:
    """Recorded selection decisions and per-kernel launch counts."""
    return {
        "decisions": list(_decisions),
        "launches": dict(_launches),
        "shadow": {op: dict(e) for op, e in _shadow.items()},
    }


def reset_census() -> None:
    _decisions.clear()
    _launches.clear()
    _shadow.clear()
