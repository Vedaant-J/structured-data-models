# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from itertools import pairwise, permutations

import pytest

from benchmark.inference.schedule import pairs, schedule, summarize

ARMS = ("eager", "compiled", "optimized")


def test_schedule_is_mirrored_and_balanced() -> None:
    blocks = schedule(ARMS, 12)
    assert len(blocks) == 12
    for block in blocks:
        assert block[3:] == block[:3][::-1]
        assert Counter(block) == dict.fromkeys(ARMS, 2)
    assert Counter(block[:3] for block in blocks) == dict.fromkeys(
        permutations(ARMS), 2
    )
    assert schedule(ARMS, 12) == blocks


def test_schedule_rejects_unbalanced_block_count() -> None:
    with pytest.raises(ValueError, match="multiple of 3"):
        schedule(ARMS, 4)


@pytest.mark.parametrize("n", [4, 5, 6])
def test_schedule_uses_a_williams_square_for_many_arms(n: int) -> None:
    arms = tuple(f"arm{i}" for i in range(n))
    blocks = schedule(arms, 2 * n)
    firsts = [block[:n] for block in blocks]
    assert len(set(firsts)) == n
    for position in range(n):
        assert Counter(row[position] for row in firsts) == dict.fromkeys(
            arms, 2
        )
    if n % 2 == 0:
        follows = Counter(p for row in set(firsts) for p in pairwise(row))
        assert set(follows.values()) == {1}
        assert len(follows) == n * (n - 1)


def test_pairs_compare_every_optimized_arm() -> None:
    arms = ("compiled", "optimized:off", "optimized:int8", "eager")
    assert pairs(arms) == [
        ("compiled", "optimized:off"),
        ("compiled", "optimized:int8"),
        ("eager", "optimized:off"),
        ("eager", "optimized:int8"),
        ("eager", "compiled"),
    ]


def test_summarize_from_raw_samples() -> None:
    seconds = {
        (0, "compiled"): [1.0, 3.0],
        (0, "optimized"): [1.0, 1.0],
        (1, "compiled"): [2.0, 2.0],
        (1, "optimized"): [3.0, 3.0],
    }
    samples = [
        {"block": block, "arm": arm, "seconds": value}
        for (block, arm), values in seconds.items()
        for value in values
    ]
    stats = summarize(samples, ("compiled", "optimized"))

    assert stats["arms"]["compiled"]["mean"] == 2.0
    assert stats["arms"]["compiled"]["median"] == 2.0
    assert stats["arms"]["optimized"]["min"] == 1.0
    assert stats["arms"]["optimized"]["stdev"] == pytest.approx(1.1547005)
    assert set(stats["pairs"]) == {"compiled_to_optimized"}
    pair = stats["pairs"]["compiled_to_optimized"]
    assert pair["reduction_pct"] == 0.0
    assert pair["block_effects"] == [1.0, -1.0]
    assert pair["median_block_effect"] == 0.0
    assert pair["positive_blocks"] == 1
