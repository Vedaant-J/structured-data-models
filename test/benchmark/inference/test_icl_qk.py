# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from benchmark.inference.kernels import registry
from sdm._kernels import rmsnorm_cast
from sdm.cache import Cache
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.nn import LogScale

SIZES = pytest.mark.parametrize(
    ("d", "heads", "kv"), [(1024, 16, 2), (512, 8, 2), (512, 8, None)]
)
TRAIN = 512


def _fit(
    d: int, heads: int, kv: int | None, rows: int, train: int = TRAIN
) -> tuple:
    torch.manual_seed(0)
    block = ICLBlock(3, 3, d, 2, heads, kv, device="cuda")

    def fn(x: Tensor, y: Tensor) -> tuple:
        cache = Cache()
        out = block(x, y, cache=cache)
        return out, [t for entry in cache.values() for t in entry]

    def make() -> tuple[Tensor, Tensor]:
        g = torch.Generator("cuda").manual_seed(0)
        x = torch.randn(
            rows, d, device="cuda", generator=g, dtype=torch.bfloat16
        )
        return x, torch.randint(0, 3, (train,), device="cuda", generator=g)

    return fn, make


@SIZES
@pytest.mark.parametrize("rows", [TRAIN, TRAIN + 96])
@pytest.mark.parametrize(
    "kernel", ["sm120_icl_query_rms_scale", "sm120_icl_key_rms"]
)
def test_icl_fit(
    matches_native: Callable,
    d: int,
    heads: int,
    kv: int | None,
    rows: int,
    kernel: str,
) -> None:
    matches_native(*_fit(d, heads, kv, rows), kernel)


@pytest.mark.parametrize(("d", "heads"), [(1024, 16), (512, 8)])
def test_icl_last_layer_narrows_the_kv_projection(
    matches_native: Callable, d: int, heads: int
) -> None:
    code = matches_native(*_fit(d, heads, 2, 1024, 1024), "sm120_icl_key_rms")
    shapes = [
        c["args"][0]["shape"]
        for c in registry.census()["decisions"]
        if c["kernel"] == "sm120_icl_key_rms"
    ]
    assert [1024, 4 * 64] in shapes, shapes
    body = code[code.index("def call(") :]
    assert "clone_split" not in body
    assert "clone_slice" not in body


@pytest.mark.parametrize("m", [16, 1000])
@pytest.mark.parametrize(("d", "heads"), [(1024, 16), (512, 8)])
def test_icl_predict_query(
    matches_native: Callable, m: int, d: int, heads: int
) -> None:
    scaling = LogScale(heads, device="cuda")

    def fn(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        q = F.linear(x, w, b).unflatten(-1, [heads, 64])
        return scaling(rmsnorm_cast(q, None, 1e-6), key_len=5000)

    def make() -> tuple[Tensor, ...]:
        g = torch.Generator("cuda").manual_seed(0)
        return tuple(
            torch.randn(*s, device="cuda", generator=g)
            for s in ((m, d), (d, d), (d,))
        )

    matches_native(fn, make, "sm120_icl_query_rms_scale")
