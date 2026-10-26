# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from benchmark.inference.kernels import registry
from benchmark.inference.kernels.registry import Kernel


@pytest.fixture(autouse=True)
def _empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "KERNELS", {})
    registry.reset_census()


def _kernel(name: str, reason: str | None = None) -> Kernel:
    return Kernel(
        op="linear_gelu",
        name=name,
        supports=lambda x, weight, bias: reason,
        impl=lambda x, weight, bias: torch.full(
            (x.size(0), weight.size(0)), 7.0
        ),
        cuda_symbols=(name,),
    )


def _args() -> tuple[torch.Tensor, ...]:
    return torch.randn(2, 3), torch.randn(4, 3), torch.randn(4)


def test_nothing_is_selected_outside_enabled() -> None:
    registry.register(_kernel("fast"))
    assert registry.select("linear_gelu", *_args()) is None
    assert registry.census() == {
        "decisions": [],
        "launches": {},
        "shadow": {},
    }


def test_first_admitting_kernel_is_selected_and_recorded() -> None:
    registry.register(_kernel("strict", reason="needs bfloat16"))
    registry.register(_kernel("fast"))
    with registry.enabled():
        for _ in range(2):
            kernel = registry.select("linear_gelu", *_args())
            out = registry.launch(kernel, *_args())
    assert torch.equal(out, torch.full((2, 4), 7.0))

    census = registry.census()
    assert census["launches"] == {"fast": 2}
    decision = census["decisions"][0]
    assert decision["kernel"] == "fast"
    assert decision["rejected"] == {"strict": "needs bfloat16"}
    assert decision["args"][0]["shape"] == [2, 3]
    assert decision["args"][0]["dtype"] == "torch.float32"

    registry.reset_census()
    assert registry.census() == {
        "decisions": [],
        "launches": {},
        "shadow": {},
    }


def test_native_when_no_kernel_admits() -> None:
    registry.register(_kernel("strict", reason="needs sm_120"))
    with registry.enabled():
        assert registry.select("linear_gelu", *_args()) is None
    census = registry.census()
    assert census["decisions"][0]["kernel"] == "native"
    assert census["decisions"][0]["rejected"] == {"strict": "needs sm_120"}
    assert census["launches"] == {}


def test_register_rejects_unknown_op_and_duplicate_name() -> None:
    registry.register(_kernel("fast"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_kernel("fast"))
    with pytest.raises(ValueError, match="unknown op"):
        registry.register(
            Kernel(
                op="missing",
                name="other",
                supports=lambda: None,
                impl=lambda: torch.zeros(1),
                cuda_symbols=(),
            )
        )


@pytest.mark.parametrize(
    ("setting", "backend", "options"),
    [
        ("sm120", "sm120", set()),
        ("off", "off", set()),
        ("sm120+uncapped", "sm120", {"uncapped"}),
        ("int8+ends", "int8", {"ends"}),
    ],
)
def test_parse_attention(setting: str, backend: str, options: set) -> None:
    assert registry.parse_attention(setting) == (backend, options)


@pytest.mark.parametrize(
    "setting",
    ["", "fp8", "sm120+ends", "int8+fast", "int8+ends+ends", "int8+uncapped"],
)
def test_parse_attention_rejects(setting: str) -> None:
    with pytest.raises(ValueError, match="ICL attention"):
        registry.parse_attention(setting)


def test_attention_setting_is_scoped_to_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(registry.ATTENTION_ENV, raising=False)
    assert registry.attention() == "sm120"
    monkeypatch.setenv(registry.ATTENTION_ENV, "off")
    with registry.enabled(attention="int8+ends"):
        assert registry.attention() == "int8+ends"
        with registry.enabled():
            assert registry.attention() == "off"
    assert registry.attention() == "off"


def test_decisions_record_lossy_kernels() -> None:
    registry.register(Kernel(**{**_kernel("lossy").__dict__, "exact": False}))
    with registry.enabled():
        registry.select("linear_gelu", *_args())
    assert not registry.census()["decisions"][0]["exact"]
