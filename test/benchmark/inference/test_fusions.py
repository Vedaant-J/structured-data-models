# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each fusion against native Inductor on small SDM-shaped inputs."""

import gc
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from benchmark.inference.kernels import registry, sm120
from benchmark.inference.kernels.fusions import (
    gating,
    kv_rms,
    linear_gelu,
    projection_rms,
    query_rms,
    rope_rms,
)
from sdm._kernels import rmsnorm_cast
from sdm.nn import GatedLogScale, RMSNorm, RotaryEmbedding

EPS = torch.finfo(torch.float32).eps
CONTEXTS, FEATURES = 64, 24
WIDTHS = pytest.mark.parametrize(("d", "head"), [(128, 32), (256, 64)])
# (K, N) of every linear_gelu kernel.
LINEAR_GELU = [*linear_gelu.MAX_M, *((h, gating.HIDDEN) for h in gating.HEADS)]


def _seeded(build: Callable[[Callable[..., Tensor]], tuple]) -> Callable:
    """Identical fresh inputs on every call, drawn by ``build``."""

    def make() -> tuple:
        g = torch.Generator("cuda").manual_seed(0)
        return build(
            lambda *shape, dtype=torch.bfloat16: torch.randn(
                *shape, device="cuda", generator=g, dtype=dtype
            )
        )

    return make


def _add(a: Tensor, b: Tensor, swapped: bool) -> Tensor:
    return b + a if swapped else a + b


def _table(
    r: Callable[..., Tensor], d: int, readouts: int = 4
) -> tuple[Tensor, Tensor]:
    """The table and its feature rows, views of one hidden flat buffer.

    AOT then passes that buffer as one flat graph input, as for SDM's table.
    """
    table = r(CONTEXTS * (FEATURES + 4) * d).view(CONTEXTS, FEATURES + 4, d)
    return table, table[:, readouts:]


@pytest.mark.parametrize(("k", "n"), LINEAR_GELU)
def test_linear_gelu(matches_native: Callable, k: int, n: int) -> None:
    def fn(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return F.gelu(F.linear(x, w, b))

    m = 2048 if k == 1024 else 4096
    matches_native(
        fn,
        _seeded(lambda r: (r(m, k), r(n, k), r(n))),
        f"sm120_linear_gelu_k{k}_n{n}",
    )


@pytest.mark.parametrize("head", [32, 64])
def test_gated_log_scale(matches_native: Callable, head: int) -> None:
    scaling = GatedLogScale(
        channels=head, num_heads=4, hidden_channels=64, device="cuda"
    )
    torch.nn.init.normal_(scaling.gate[-1].weight, std=0.125)

    def fn(q: Tensor) -> Tensor:
        return scaling(q, key_len=q.size(-3))

    matches_native(
        fn, _seeded(lambda r: (r(16, 32, 4, head),)), "sm120_gated_log_scale"
    )


@WIDTHS
def test_rope_rms(matches_native: Callable, d: int, head: int) -> None:
    rope = RotaryEmbedding(
        head, "split_half", requires_grad=False, device="cuda"
    )
    norm = RMSNorm(head, eps=1e-6, elementwise_affine=False, device="cuda")

    def fn(q: Tensor, kv: Tensor) -> tuple[Tensor, Tensor]:
        k = kv[..., :d].unflatten(-1, [4, head])
        return norm(q.unflatten(-1, [4, head]), rope=rope), norm(k, rope=rope)

    matches_native(
        fn,
        _seeded(lambda r: (r(50, 57, d), r(50, 57, 2 * d))),
        "sm120_rope_rms",
    )


@WIDTHS
def test_rope_rms_uneven(matches_native: Callable, d: int, head: int) -> None:
    rope = RotaryEmbedding(
        head, "split_half", requires_grad=False, device="cuda"
    )
    norm = RMSNorm(head, eps=1e-6, elementwise_affine=False, device="cuda")

    def fn(q: Tensor, kv: Tensor) -> tuple[Tensor, Tensor]:
        k = kv[..., :d].unflatten(-1, [4, head])
        return norm(q.unflatten(-1, [4, head]), rope=rope), norm(k, rope=rope)

    matches_native(
        fn,
        _seeded(lambda r: (r(50, 4, d), r(50, 57, 2 * d))),
        "sm120_rope_rms_uneven",
    )


@WIDTHS
def test_linear_kv_key_rms(
    matches_native: Callable, d: int, head: int
) -> None:
    norm = RMSNorm(head, eps=1e-6, elementwise_affine=False, device="cuda")

    def fn(x: Tensor, w: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        k, v = F.linear(x, w, b).chunk(2, -1)
        return norm(k.unflatten(-1, [4, head])), v

    matches_native(
        fn,
        _seeded(lambda r: (r(FEATURES, CONTEXTS, d), r(2 * d, d), r(2 * d))),
        "sm120_linear_kv_key_rms",
    )


@WIDTHS
def test_linear_query_rms(matches_native: Callable, d: int, head: int) -> None:
    norm = RMSNorm(head, eps=1e-6, elementwise_affine=False, device="cuda")

    def fn(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return norm(F.linear(x, w, b).unflatten(-1, [4, head]))

    matches_native(
        fn,
        _seeded(lambda r: (r(FEATURES, CONTEXTS, d), r(d, d), r(d))),
        "sm120_linear_query_rms",
    )


def _column_projection(d: int, swapped: bool) -> tuple[Callable, Callable]:
    def fn(t: Tensor, rows: Tensor, x: Tensor, *params: Tensor) -> tuple:
        w, b, norm_weight = params
        raw = F.linear(x, w, b)
        prior = rows.transpose(0, 1)
        residual = _add(raw.view(prior.shape), prior, swapped)
        out = rmsnorm_cast(residual, norm_weight, EPS)
        t[:, :4] = 0
        return raw, out

    make = _seeded(
        lambda r: (
            *_table(r, d),
            r(FEATURES * CONTEXTS, d),
            r(d, d),
            r(d),
            r(d, dtype=torch.float32),
        )
    )
    return fn, make


@WIDTHS
def test_column_projection_rms(
    matches_native: Callable, d: int, head: int
) -> None:
    matches_native(
        *_column_projection(d, swapped=False), "sm120_column_projection_rms"
    )


@WIDTHS
def test_column_projection_rms_skips_a_transposed_sum(
    skips_rule: Callable, d: int, head: int
) -> None:
    """A transposed prior + raw gets cloned before the norm; stay native."""
    skips_rule(*_column_projection(d, swapped=True), "column_projection_rms")


D = 128


def test_gated_log_scale_needs_four_heads(skips_rule: Callable) -> None:
    scaling = GatedLogScale(
        channels=32, num_heads=2, hidden_channels=64, device="cuda"
    )
    torch.nn.init.normal_(scaling.gate[-1].weight, std=0.125)

    def fn(q: Tensor) -> Tensor:
        return scaling(q, key_len=q.size(-3))

    skips_rule(fn, _seeded(lambda r: (r(32, 32, 2, 32),)), "gated_log_scale")


def test_linear_kv_key_rms_keeps_a_read_raw_key(skips_rule: Callable) -> None:
    norm = RMSNorm(32, eps=1e-6, elementwise_affine=False, device="cuda")

    def fn(x: Tensor, w: Tensor, b: Tensor) -> tuple[Tensor, ...]:
        k, v = F.linear(x, w, b).chunk(2, -1)
        return norm(k.unflatten(-1, [4, 32])), v, k * 2

    skips_rule(
        fn,
        _seeded(lambda r: (r(FEATURES, CONTEXTS, D), r(2 * D, D), r(2 * D))),
        "linear_kv_key_rms",
    )


def test_linear_query_rms_keeps_a_read_raw_query(skips_rule: Callable) -> None:
    norm = RMSNorm(32, eps=1e-6, elementwise_affine=False, device="cuda")

    def fn(x: Tensor, w: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        q = F.linear(x, w, b)
        return norm(q.unflatten(-1, [4, 32])), q * 2

    skips_rule(
        fn,
        _seeded(lambda r: (r(FEATURES, CONTEXTS, D), r(D, D), r(D))),
        "linear_query_rms",
    )


def test_column_projection_rms_needs_four_readouts(
    skips_rule: Callable,
) -> None:
    def fn(t: Tensor, rows: Tensor, x: Tensor, *params: Tensor) -> tuple:
        w, b, norm_weight = params
        raw = F.linear(x, w, b)
        prior = rows.transpose(0, 1)
        out = rmsnorm_cast(raw.view(prior.shape) + prior, norm_weight, EPS)
        t[:, :3] = 0
        return raw, out

    skips_rule(
        fn,
        _seeded(
            lambda r: (
                *_table(r, D, readouts=3),
                r((FEATURES + 1) * CONTEXTS, D),
                r(D, D),
                r(D),
                r(D, dtype=torch.float32),
            )
        ),
        "column_projection_rms",
    )


def _empty(*shape: int, dtype: torch.dtype = torch.bfloat16) -> Tensor:
    return torch.empty(*shape, device="cuda", dtype=dtype)


def _f32(*shape: int) -> Tensor:
    return _empty(*shape, dtype=torch.float32)


REJECTED = {
    "sm120_linear_gelu_k32_n64": (
        lambda: (_empty(512, 32), _empty(64, 32), _empty(64)),
        "needs x of shape [1024<=M",
    ),
    "sm120_linear_gelu_k128_n256": (
        lambda: (_empty(4096, 128), _empty(128, 128), _empty(128)),
        "needs a 256x128 weight",
    ),
    "sm120_gated_log_scale": (
        lambda: (
            _empty(2048, 64),
            _empty(48, 64),
            _empty(48),
            _empty(2048, 48),
            _f32(4),
            1.0,
        ),
        "with head in (32, 64)",
    ),
    "sm120_rope_rms": (
        lambda: (_empty(1, 2048, 4, 32), _empty(1, 2048, 4, 32), _f32(16)),
        "needs 1 <= S <= 1024",
    ),
    "sm120_linear_kv_key_rms": (
        lambda: (_empty(128, 256), _empty(512, 256), _empty(512), 64),
        "needs 129 <= M",
    ),
    "sm120_linear_query_rms": (
        lambda: (_empty(1000, 128), _empty(128, 128), _empty(128), 32),
        "differs from cuBLAS",
    ),
    "sm120_column_projection_rms": (
        lambda: (
            _empty(CONTEXTS * 25, 128),
            _empty(128, 128),
            _empty(128),
            _empty(CONTEXTS, 28, 128),
            _f32(128),
            EPS,
        ),
        "a table of 4 + F rows",
    ),
    "sm120_rope_rms_uneven": (
        lambda: (_empty(8, 57, 4, 64), _empty(8, 57, 4, 64), _f32(32)),
        "different S",
    ),
}


@pytest.mark.parametrize("kernel", sorted(REJECTED))
def test_supports_rejects(rejects: Callable, kernel: str) -> None:
    make, reason = REJECTED[kernel]
    rejects(kernel, *make(), reason=reason)


def _rows(m: int, *widths: int) -> list[Tensor]:
    """``[m, w]`` views of one buffer; ``supports`` reads only metadata."""
    flat = _empty(m * max(widths))
    return [flat[: m * w].view(m, w) for w in widths]


def _column(m: int, d: int) -> tuple[Tensor, Tensor]:
    """Raw ``[C * F, D]`` and table ``[C, 4 + F, D]`` at C = 1."""
    (rows,) = _rows(m + 4, d)
    return rows[:m], rows.view(1, m + 4, d)


def _linear_gelu_args(m: int, k: int, n: int) -> tuple:
    return *_rows(m, k), _empty(n, k), _empty(n)


def _gls_args(m: int, head: int) -> tuple:
    x, query = _rows(m, 64, head)
    return x, _empty(head, 64), _empty(head), query, _f32(4), 1.0


def _rope_args(m: int, head: int) -> tuple:
    (q,) = _rows(m, 4 * head)
    q = q.view(m, 1, 4, head)
    return q, q, _f32(head // 2)


def _column_projection_args(m: int, d: int) -> tuple:
    x, table = _column(m, d)
    return x, _empty(d, d), _empty(d), table, _f32(d), EPS


D_HEAD = ((128, 32), (256, 64))
# kernel -> [(largest checked M of one instance, M -> supports() args)]
ROW_LIMITS = {
    **{
        f"sm120_linear_gelu_k{k}_n{n}": [
            (limit, lambda m, k=k, n=n: _linear_gelu_args(m, k, n))
        ]
        for (k, n), limit in (
            *linear_gelu.MAX_M.items(),
            *(((h, gating.HIDDEN), m) for h, m in gating.FIRST_MAX_M.items()),
        )
    },
    "sm120_gated_log_scale": [
        (gating.SECOND_MAX_M[h], lambda m, h=h: _gls_args(m, h))
        for h in (32, 64)
    ],
    "sm120_linear_kv_key_rms": [
        (
            kv_rms.MAX_M[d],
            lambda m, d=d, h=h: (
                *_rows(m, d),
                _empty(2 * d, d),
                _empty(2 * d),
                h,
            ),
        )
        for d, h in D_HEAD
    ],
    "sm120_linear_query_rms": [
        (
            query_rms.MAX_M[d],
            lambda m, d=d, h=h: (*_rows(m, d), _empty(d, d), _empty(d), h),
        )
        for d, h in D_HEAD
    ],
    "sm120_rope_rms": [
        (rope_rms.MAX_TOKENS[h], lambda m, h=h: _rope_args(m, h))
        for _, h in D_HEAD
    ],
    "sm120_column_projection_rms": [
        (
            projection_rms.MAX_M[d],
            lambda m, d=d: _column_projection_args(m, d),
        )
        for d, _ in D_HEAD
    ],
}


@pytest.mark.parametrize(
    ("kernel", "instance"),
    [(k, i) for k in sorted(ROW_LIMITS) for i in range(len(ROW_LIMITS[k]))],
)
def test_row_limits(rejects: Callable, kernel: str, instance: int) -> None:
    """The largest checked row count is admitted; one more is refused."""
    limit, make = ROW_LIMITS[kernel][instance]
    for m in (limit, limit + 1):
        args = make(m)
        if m == limit:
            assert registry.kernel(kernel).supports(*args) is None
        else:
            rejects(kernel, *args, reason=str(limit))
        del args
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize(("k", "n"), LINEAR_GELU)
def test_linear_gelu_leaves_query_side_calls_native(
    rejects: Callable, k: int, n: int
) -> None:
    kernel = f"sm120_linear_gelu_k{k}_n{n}"
    for m in (sm120.MIN_M, sm120.MIN_M - 1):
        args = _linear_gelu_args(m, k, n)
        if m == sm120.MIN_M:
            assert registry.kernel(kernel).supports(*args) is None
        else:
            rejects(kernel, *args, reason=f"{sm120.MIN_M}<=M")
