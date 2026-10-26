# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from benchmark.inference.kernels import fx_pass, registry
from benchmark.inference.kernels.registry import Kernel

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)


@pytest.fixture(autouse=True)
def _empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "KERNELS", {})
    registry.reset_census()


def _mlp(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return F.gelu(F.linear(x, w, b)).reshape(2, -1, w.size(0)).sum(0)


def _tanh_mlp(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return F.gelu(F.linear(x, w, b), approximate="tanh")


def _run(
    reason: str | None, fn: Callable[..., Tensor] = _mlp
) -> tuple[Tensor, Tensor]:
    registry.register(
        Kernel(
            op="linear_gelu",
            name="test_kernel",
            supports=lambda x, w, b: reason,
            impl=lambda x, w, b: F.gelu(F.linear(x, w, b)),
            cuda_symbols=("test",),
        )
    )
    torch._dynamo.reset()
    compiled = torch.compile(fn, options=fx_pass.compile_options())
    args = tuple(
        torch.randn(*shape, dtype=torch.bfloat16)
        for shape in ((8, 16), (32, 16), (32,))
    )
    with registry.enabled():
        return compiled(*args), fn(*args)


def test_admitted_region_runs_the_selected_kernel() -> None:
    out, expected = _run(None)
    torch.testing.assert_close(out, expected)
    census = registry.census()
    assert [d["kernel"] for d in census["decisions"]] == ["test_kernel"]
    assert census["launches"] == {"test_kernel": 1}


def test_rejected_region_stays_native() -> None:
    _run("unsupported shape")
    census = registry.census()
    assert census["decisions"][0]["kernel"] == "native"
    assert census["decisions"][0]["rejected"] == {
        "test_kernel": "unsupported shape"
    }
    assert census["launches"] == {}


def test_tanh_gelu_is_not_rewritten() -> None:
    out, expected = _run(None, _tanh_mlp)
    torch.testing.assert_close(out, expected)
    assert _selected() == []


def test_rules_for_one_target_run_in_order_until_one_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(fx_pass, "RULES", [])
    erf = torch.ops.aten.erf.default
    fx_pass.add_rule(erf, "a", lambda g, n: seen.append("a"))
    fx_pass.add_rule(erf, "b", lambda g, n: seen.append("b") or True)
    fx_pass.add_rule(erf, "c", lambda g, n: seen.append("c"))
    torch._dynamo.reset()
    fn = torch.compile(torch.erf, options=fx_pass.compile_options())
    fn(torch.randn(4))
    assert seen == ["a", "b"]


@needs_cuda
def test_triton_kernel_node_is_matchable_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")

    @triton.jit
    def _double_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + i, tl.load(x_ptr + i, mask=i < n) * 2, mask=i < n)

    def double(x: Tensor) -> Tensor:
        out = torch.empty_like(x)
        _double_kernel[(1,)](x, out, x.numel(), BLOCK=64)
        return out

    names: list[str | None] = []
    monkeypatch.setattr(fx_pass, "RULES", [])
    fx_pass.add_rule(
        fx_pass.TRITON_KERNEL,
        "double",
        lambda g, n: names.append(fx_pass.triton_kernel_name(n)),
    )
    torch._dynamo.reset()
    torch.compile(double, options=fx_pass.compile_options())(
        torch.randn(64, device="cuda")
    )
    assert names == ["_double_kernel"]


def _selected() -> list[str]:
    return [
        d["kernel"]
        for d in registry.census()["decisions"]
        if d["kernel"] != "native"
    ]


def test_shadow_keeps_native_outputs_and_records_the_kernel() -> None:
    registry.register(
        Kernel(
            op="linear_gelu",
            name="test_kernel",
            supports=lambda x, w, b: None,
            impl=lambda x, w, b: F.gelu(F.linear(x, w, b)) + 1,
            cuda_symbols=("test",),
        )
    )
    args = tuple(
        torch.randn(*shape, dtype=torch.bfloat16)
        for shape in ((8, 16), (32, 16), (32,))
    )
    torch._dynamo.reset()
    native = torch.compile(_mlp)(*args)
    torch._dynamo.reset()
    shadow = torch.compile(_mlp, options=fx_pass.compile_options(shadow=True))
    with registry.enabled():
        out = shadow(*args)
    # Native stays on the data path (the kernel adds 1); Inductor may still
    # fuse it differently around the comparison.
    assert (out.float() - native.float()).abs().max() < 0.5
    entry = registry.census()["shadow"]["linear_gelu"]
    assert (entry["outputs"], entry["bitwise"]) == (1, 0)
    assert entry["max_abs_diff"] > 0.5
