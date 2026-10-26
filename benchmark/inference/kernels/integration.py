# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from collections.abc import Callable
from pathlib import Path

import torch

from benchmark.inference.kernels import build, fusions, fx_pass, registry


def default_cache() -> Path:
    root = os.environ.get("SDM_KERNEL_CACHE")
    if root:
        return Path(root)
    version = f"{torch.__version__}-{torch.version.cuda}"
    return Path.home() / ".cache" / "sdm-inference-kernels" / version


class Unsupported(RuntimeError):
    """The kernels cannot run on this machine; the native arms still can."""


NOT_BITWISE = frozenset({"icl_attention"})


def install(
    kernels: Path | None,
    ops: tuple[str, ...] | None = None,
    exact: bool = False,
    shadow: bool = False,
) -> dict:
    """Register the kernels, building missing ones; return Inductor options.

    ``kernels`` defaults to :func:`default_cache`. ``exact`` leaves the ops in
    :data:`NOT_BITWISE` native.
    """
    if torch.cuda.get_device_capability() != (12, 0):
        raise Unsupported(
            "the kernels need an SM 12.0 (Blackwell) GPU, found "
            f"{torch.cuda.get_device_name()}"
        )
    if not registry.registered():
        fusions.load(build.build(kernels or default_cache(), verbose=True))
    unknown = set(ops or ()) - {op for _, op, _ in fx_pass.RULES}
    if unknown:
        raise RuntimeError(f"no graph rule for ops {sorted(unknown)}")
    if exact:
        every = {op for _, op, _ in fx_pass.RULES}
        ops = tuple(set(ops or every) - NOT_BITWISE)
    return fx_pass.compile_options(
        None if ops is None else frozenset(ops), shadow
    )


COMPILE = {
    "fullgraph": False,
    "dynamic": False,
    "options": {"triton.cudagraphs": False},
}


def isolate_compiles() -> None:
    """Keep every ``torch.compile`` of a process on its own fresh graph."""
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    torch._inductor.config.fx_graph_cache = False
    torch._functorch.config.enable_autograd_cache = False


def compile_with_kernels(
    fn: Callable,
    kernels: Path | None = None,
    ops: tuple[str, ...] | None = None,
    exact: bool = False,
    attention: str | None = None,
    shadow: bool = False,
) -> Callable:
    """``torch.compile(fn)`` with the kernels routed in and enabled.

    ``attention`` overrides ``$SDM_ICL_ATTENTION`` (see
    :func:`registry.parse_attention`). ``shadow`` runs every kernel beside
    native and records whether it reproduced native bit for bit
    (:func:`fx_pass.replace`); the outputs stay native.
    """
    if attention is not None:
        registry.parse_attention(attention)
    options = install(kernels, ops, exact, shadow)
    compiled = torch.compile(
        fn, **{**COMPILE, "options": {**COMPILE["options"], **options}}
    )

    @functools.wraps(fn)
    def run(*args, **kwargs):
        with registry.enabled(attention):
            return compiled(*args, **kwargs)

    return run


def enable(
    model: torch.nn.Module,
    kernels: Path | None = None,
    ops: tuple[str, ...] | None = None,
    exact: bool = False,
    attention: str | None = None,
) -> None:
    """Compile every task model of an SDM ``model`` with the kernels.

    ``exact=True`` keeps ICL attention native, so the outputs stay bitwise
    equal to compiled SDM. ``attention`` (default ``$SDM_ICL_ATTENTION``,
    else ``sm120``) picks the ICL attention kernel, e.g. ``int8``.
    Raises :class:`Unsupported` on a GPU the kernels do not support. Undo with
    :func:`disable`.
    """
    for inner in model.models.values():
        inner.forward = compile_with_kernels(
            inner.forward, kernels, ops, exact, attention
        )


def disable(model: torch.nn.Module) -> None:
    for inner in model.models.values():
        vars(inner).pop("forward", None)
