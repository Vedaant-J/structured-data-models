# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch._inductor.utils import run_and_get_code
from torch.fx import Graph

from benchmark.inference.kernels import fx_pass, integration, registry
from benchmark.inference.kernels.fusions import icl_attention
from benchmark.inference.kernels.registry import Kernel

HEADS, HEAD = 8, 64
KERNELS = {
    "sm120": "sm120_icl_attention_d64",
    "fa2": "fa2_icl_attention_d64",
    "int8": "sm120_int8_icl",
}
ATOL = {"sm120": 2e-2, "fa2": 2e-2, "int8": 8e-2}


def _seeded(queries: int, keys: int, batch: int = 1) -> Callable[[], tuple]:
    def make() -> tuple:
        g = torch.Generator("cuda").manual_seed(0)
        return tuple(
            torch.randn(
                batch,
                n,
                HEADS,
                HEAD,
                device="cuda",
                generator=g,
                dtype=torch.bfloat16,
            )
            for n in (queries, keys, keys)
        )

    return make


def _attend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


@pytest.fixture(params=sorted(KERNELS))
def mode(request, monkeypatch, kernels: None) -> str:
    if request.param == "fa2" and icl_attention._flash_attn_func() is None:
        pytest.skip("flash_attn is not importable")
    monkeypatch.setenv(icl_attention.ENV, request.param)
    return request.param


@pytest.mark.parametrize("batch", [1, 8])
def test_fit_attention_is_replaced(
    close_to_native: Callable, mode: str, batch: int
) -> None:
    make = _seeded(2048, 2048, batch)
    code = close_to_native(_attend, make, KERNELS[mode], atol=ATOL[mode])
    assert "sdm_inference.icl_attention.default(" in code
    q, k, v = make()
    reference = _attend(q.float(), k.float(), v.float())
    kernel = registry.kernel(KERNELS[mode])
    with torch.inference_mode():
        out = registry.launch(kernel, q, k, v, HEAD**-0.5)
    assert out.shape == q.shape
    assert (out.float() - reference).abs().max().item() <= ATOL[mode]


def test_predict_shape_stays_native(stays_native: Callable, mode: str) -> None:
    stays_native(_attend, _seeded(16, 2048), "icl_attention")


def test_off_stays_native(
    stays_native: Callable, monkeypatch, kernels: None
) -> None:
    monkeypatch.setenv(icl_attention.ENV, "off")
    stays_native(_attend, _seeded(2048, 2048), "icl_attention")


@pytest.mark.parametrize(
    ("rows", "admitted"), [(1023, False), (65_535, True), (65_536, False)]
)
def test_bf16_row_bounds(
    monkeypatch, kernels: None, rows: int, admitted: bool
) -> None:
    monkeypatch.setenv(icl_attention.ENV, "sm120")
    t = torch.empty(1, rows, HEADS, HEAD, device="cuda", dtype=torch.bfloat16)
    kernel = registry.kernel(KERNELS["sm120"])
    assert (kernel.supports(t, t, t, HEAD**-0.5) is None) == admitted


def test_uncapped_bf16_admits_contexts_above_65535(
    monkeypatch, kernels: None
) -> None:
    monkeypatch.setenv(icl_attention.ENV, "sm120+uncapped")
    t = torch.empty(
        1, 65_536, HEADS, HEAD, device="cuda", dtype=torch.bfloat16
    )
    kernel = registry.kernel(KERNELS["sm120"])
    assert kernel.supports(t, t, t, HEAD**-0.5) is None


def test_int8_admits_contexts_above_65535(monkeypatch, kernels: None) -> None:
    monkeypatch.setenv(icl_attention.ENV, "int8")
    t = torch.empty(
        1, 100_000, HEADS, HEAD, device="cuda", dtype=torch.bfloat16
    )
    assert registry.kernel(KERNELS["int8"]).supports(t, t, t, 0.125) is None


def test_int8_batch_bounds(monkeypatch, kernels: None) -> None:
    monkeypatch.setenv(icl_attention.ENV, "int8")
    t = torch.empty(1, 2048, HEADS, HEAD, device="cuda", dtype=torch.bfloat16)
    kernel = registry.kernel(KERNELS["int8"])
    assert kernel.supports(*(t.expand(65_535, -1, -1, -1),) * 3, 0.125) is None
    wide = t.expand(65_536, -1, -1, -1)
    assert "batch and heads" in kernel.supports(wide, wide, wide, 0.125)
    long = t[:, :1].expand(8, 2**25, -1, -1)
    assert "padded S" in kernel.supports(long, long, long, 0.125)


@pytest.mark.parametrize(
    ("batch", "rows"), [(2, 1024), (8, 3001), (12, 20_000)]
)
def test_int8_batch_equals_each_entry_alone(
    kernels: None, batch: int, rows: int
) -> None:
    g = torch.Generator("cuda").manual_seed(batch)
    qkv = torch.randn(
        batch,
        rows,
        3 * HEADS * HEAD,
        device="cuda",
        generator=g,
        dtype=torch.bfloat16,
    )
    q, k, v = (t.unflatten(-1, (HEADS, HEAD)) for t in qkv.chunk(3, -1))
    impl = registry.kernel(KERNELS["int8"]).impl
    with torch.inference_mode():
        out = impl(q, k, v, HEAD**-0.5)
        for i in range(batch):
            alone = impl(q[i : i + 1], k[i : i + 1], v[i : i + 1], HEAD**-0.5)
            assert torch.equal(out[i : i + 1], alone), i


@pytest.mark.parametrize("exact", [False, True])
def test_exact_keeps_attention_native(
    monkeypatch, kernels: None, exact: bool
) -> None:
    monkeypatch.setenv(icl_attention.ENV, "sm120")
    torch._dynamo.reset()
    fn = integration.compile_with_kernels(_attend, exact=exact)
    with torch.inference_mode():
        _, codes = run_and_get_code(fn, *_seeded(2048, 2048)())
    assert ("sdm_inference.icl_attention.default(" in codes[-1]) != exact


def test_split_merge_is_deterministic(monkeypatch, kernels: None) -> None:
    monkeypatch.setenv(icl_attention.ENV, "sm120")
    q, k, v = _seeded(20000, 20000)()
    kernel = registry.kernel(KERNELS["sm120"])
    with torch.inference_mode():
        first = registry.launch(kernel, q, k, v, HEAD**-0.5)
        for _ in range(9):
            out = registry.launch(kernel, q, k, v, HEAD**-0.5)
            assert torch.equal(out, first)


class _Graph:
    """``count`` flash calls with the shapes of the ICL fit site."""

    def __init__(self, count: int, batch: int = 1) -> None:
        self.graph = Graph()
        like = torch.empty(
            batch, 8, 1024, 64, dtype=torch.bfloat16, device="meta"
        )
        inputs = []
        for name in "qkv":
            inputs.append(self.graph.placeholder(name))
            inputs[-1].meta["val"] = like
        items = []
        for _ in range(count):
            call = self.graph.call_function(
                torch.ops.aten._scaled_dot_product_flash_attention.default,
                tuple(inputs),
                {"scale": 0.125},
            )
            items.append(self.graph.call_function(operator.getitem, (call, 0)))
            items[-1].meta["val"] = like
        self.graph.output(tuple(items))

    def rewritten(self, setting: str) -> int:
        with registry.enabled(attention=setting):
            fx_pass.KernelPass(frozenset({"icl_attention"}))(self.graph)
        target = torch.ops.sdm_inference.icl_attention.default
        return sum(n.target == target for n in self.graph.nodes)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize(
    ("setting", "rewritten"),
    [("off", 0), ("int8", 4), ("int8+ends", 2)],
)
def test_rule_follows_the_setting(
    monkeypatch, setting: str, rewritten: int, batch: int
) -> None:
    monkeypatch.setattr(registry, "KERNELS", {})
    registry.register(
        Kernel(
            op="icl_attention",
            name="int8",
            supports=lambda q, k, v, scale: (
                None
                if registry.attention().replace("+ends", "") == "int8"
                else "other setting"
            ),
            impl=lambda q, k, v, scale: q,
            cuda_symbols=("int8",),
            exact=False,
        )
    )
    assert _Graph(4, batch).rewritten(setting) == rewritten


def _ops(options: dict) -> set[str]:
    rules = options["post_grad_custom_post_pass"].rules.values()
    return {op for pairs in rules for op, _ in pairs}


def test_exact_leaves_attention_native(kernels: None) -> None:
    exact = _ops(integration.install(None, exact=True))
    assert exact
    assert not exact & integration.NOT_BITWISE
