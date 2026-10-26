# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures checking real SM120 kernels against native Inductor, bitwise.

Each check compiles a small function reproducing an SDM pattern twice: plain
(the native result) and with the kernel pass enabled, so the rule, the op and
the kernel are all exercised.
"""

from collections.abc import Callable
from contextlib import nullcontext

import pytest
import torch
from torch import Tensor
from torch._inductor.utils import run_and_get_code

# Defines the linear_gelu op and rule that the unit tests give fake kernels.
import benchmark.inference.kernels.fusions.linear_gelu  # noqa: F401
from benchmark.inference.kernels import fx_pass, integration, registry
from benchmark.inference.runner import set_precision


@pytest.fixture(scope="session")
def kernels() -> None:
    """Build and register every kernel; skip without an SM 12.0 GPU."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
        12,
        0,
    ):
        pytest.skip("needs an SM 12.0 (Blackwell) GPU")
    set_precision("ieee")
    integration.install(None)


def _bits(value: Tensor) -> Tensor:
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


def _tensors(value: object) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _tensors(v)]
    return []


def _run(
    fn: Callable, make_args: Callable[[], tuple], options: dict, enabled: bool
) -> tuple[list[Tensor], str]:
    args = make_args()
    torch._dynamo.reset()
    compiled = torch.compile(
        fn,
        **{
            **integration.COMPILE,
            "options": {**integration.COMPILE["options"], **options},
        },
    )
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        registry.enabled() if enabled else nullcontext(),
    ):
        out, codes = run_and_get_code(compiled, *args)
    torch.cuda.synchronize()
    return _tensors(out) + _tensors(args), codes[-1]


def _with_kernels(
    fn: Callable, make_args: Callable[[], tuple], atol: float | None = None
) -> tuple[dict, str]:
    """Census and fused code of ``fn``, whose results match native bitwise.

    With ``atol`` they match within that max absolute difference instead.

    ``make_args`` must return identical fresh inputs on every call, so a
    mutated input (an in-place table) starts from the same values on both
    paths; outputs and all inputs are compared after the call.
    """
    registry.reset_census()
    native, _ = _run(fn, make_args, {}, enabled=False)
    fused, code = _run(fn, make_args, fx_pass.compile_options(), enabled=True)
    census = registry.census()
    assert len(native) == len(fused)
    for i, (a, b) in enumerate(zip(native, fused)):
        assert a.shape == b.shape, i
        assert a.dtype == b.dtype, i
        if atol is None:
            assert torch.equal(_bits(a), _bits(b)), f"tensor {i} differs"
        else:
            diff = (a.float() - b.float()).abs().max().item()
            assert diff <= atol, f"tensor {i} differs by {diff}"
    return census, code


def assert_kernel_matches_native(
    fn: Callable, make_args: Callable[[], tuple], kernel: str
) -> str:
    """``kernel`` runs inside ``fn`` and changes no bit of its results.

    Returns the generated code of the run with kernels.
    """
    census, code = _with_kernels(fn, make_args)
    assert kernel in {d["kernel"] for d in census["decisions"]}, census
    assert census["launches"].get(kernel, 0) > 0, census["launches"]
    return code


def assert_kernel_close_to_native(
    fn: Callable, make_args: Callable[[], tuple], kernel: str, atol: float
) -> str:
    """:func:`assert_kernel_matches_native`, within ``atol``, not bitwise."""
    census, code = _with_kernels(fn, make_args, atol)
    assert kernel in {d["kernel"] for d in census["decisions"]}, census
    assert census["launches"].get(kernel, 0) > 0, census["launches"]
    return code


def assert_stays_native(
    fn: Callable, make_args: Callable[[], tuple], op: str
) -> None:
    """``op``'s rule is tried but no kernel is chosen; results stay bitwise."""
    census, _ = _with_kernels(fn, make_args)
    chosen = [d for d in census["decisions"] if d["op"] == op]
    assert chosen, census
    assert all(d["kernel"] == "native" for d in chosen), census


def assert_rule_skips(
    fn: Callable, make_args: Callable[[], tuple], op: str
) -> None:
    """``op``'s rule leaves ``fn`` alone; any other kernel stays exact."""
    census, _ = _with_kernels(fn, make_args)
    assert all(d["op"] != op for d in census["decisions"]), census


def assert_kernel_rejects(
    kernel: str, *args: object, reason: str | None = None
) -> None:
    """``kernel``'s ``supports`` refuses ``args``, naming ``reason``."""
    refused = registry.kernel(kernel).supports(*args)
    assert refused is not None, f"{kernel} admits the arguments"
    assert reason is None or reason in refused, refused


@pytest.fixture
def matches_native(kernels: None) -> Callable:
    """:func:`assert_kernel_matches_native`, on an SM 12.0 GPU only."""
    return assert_kernel_matches_native


@pytest.fixture
def close_to_native(kernels: None) -> Callable:
    """:func:`assert_kernel_close_to_native`, on an SM 12.0 GPU only."""
    return assert_kernel_close_to_native


@pytest.fixture
def stays_native(kernels: None) -> Callable:
    """:func:`assert_stays_native`, on an SM 12.0 GPU only."""
    return assert_stays_native


@pytest.fixture
def skips_rule(kernels: None) -> Callable:
    """:func:`assert_rule_skips`, on an SM 12.0 GPU only."""
    return assert_rule_skips


@pytest.fixture
def rejects(kernels: None) -> Callable:
    """:func:`assert_kernel_rejects`, on an SM 12.0 GPU only."""
    return assert_kernel_rejects
