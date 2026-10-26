# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pyarrow as pa
import pytest
import torch

from benchmark.inference.snapshot import characterize, compare, leaves
from sdm import StringTensor


@dataclass
class _State:
    weight: torch.Tensor
    name: str


def test_leaves_walks_nested_containers() -> None:
    tree = {
        "a": (torch.ones(2), [torch.zeros(1)]),
        3: _State(torch.ones(1), "x"),
    }
    assert set(leaves(tree, "cache")) == {
        "cache/a/0",
        "cache/a/1/0",
        "cache/3/weight",
    }


def test_compare_is_bitwise() -> None:
    a = {"p": torch.tensor([0.0, float("nan"), 1.0], dtype=torch.bfloat16)}
    assert compare(a, {"p": a["p"].clone()})["bitwise_equal"]
    signed_zero = {"p": torch.tensor([-0.0, float("nan"), 1.0]).bfloat16()}
    assert not compare(a, signed_zero)["bitwise_equal"]


def test_compare_reports_first_mismatch() -> None:
    reference = {"x": torch.ones(4), "y": torch.arange(4), "z": torch.ones(2)}
    candidate = {
        "x": torch.ones(4),
        "y": torch.tensor([0, 1, 5, 7]),
        "z": torch.ones(3),
    }
    result = compare(reference, candidate)
    assert not result["bitwise_equal"]
    assert result["mismatched_leaves"] == 2
    assert result["first_mismatch"] == {
        "path": "y",
        "differing_elements": 2,
        "elements": 4,
        "max_abs_diff": 4.0,
    }
    missing = compare(reference, {"x": torch.ones(4)})
    assert missing["first_mismatch"]["reason"] == "missing in candidate"


def test_characterize_probability_distance() -> None:
    p = torch.tensor([[0.9, 0.1], [0.4, 0.6]])
    q = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    result = characterize({"probabilities": p}, {"probabilities": q})
    assert result["probability_max_abs_diff"] == pytest.approx(0.2)
    assert result["probability_mean_abs_diff"] == pytest.approx(0.15)
    assert result["argmax_agreement_pct"] == 50.0


def test_string_tensor_leaves_compare_bitwise() -> None:
    strings = StringTensor.from_arrow(pa.array(["a", "bc", "def"]))
    other = StringTensor.from_arrow(pa.array(["a", "bc", "deg"]))
    ref = leaves({"classes": strings}, "cache")
    assert ref
    assert compare(ref, leaves({"classes": strings}, "cache"))["bitwise_equal"]
    assert not compare(ref, leaves({"classes": other}, "cache"))[
        "bitwise_equal"
    ]
