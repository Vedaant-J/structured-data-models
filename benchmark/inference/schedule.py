# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import random
import statistics

SHUFFLE_SEED = 2026101507


def is_optimized(arm: str) -> bool:
    return arm == "optimized" or arm.startswith("optimized:")


def pairs(arms: tuple[str, ...]) -> list[tuple[str, str]]:
    """(base, candidate) arm pairs compared in the statistics."""
    optimized = [arm for arm in arms if is_optimized(arm)]
    bases = [base for base in ("compiled", "eager") if base in arms]
    result = [(base, arm) for base in bases for arm in optimized]
    if len(bases) == 2:
        result.append(("eager", "compiled"))
    return result


def _williams_square(n: int) -> list[tuple[int, ...]]:
    """Latin square; for even ``n`` each index follows every other once."""
    first, low, high = [0], 1, n - 1
    while len(first) < n:
        first.append(low)
        low += 1
        if len(first) < n:
            first.append(high)
            high -= 1
    return [tuple((x + i) % n for x in first) for i in range(n)]


def orders(arms: tuple[str, ...], blocks: int) -> list[tuple[str, ...]]:
    """Every permutation if ``blocks`` allows, else a Williams square."""
    every = list(itertools.permutations(arms))
    if blocks % len(every) == 0:
        return every
    if blocks % len(arms) == 0:
        rows = _williams_square(len(arms))
        return [tuple(arms[i] for i in row) for row in rows]
    raise ValueError(
        f"blocks must be a multiple of {len(arms)} (or {len(every)} for "
        "every order)"
    )


def schedule(arms: tuple[str, ...], blocks: int) -> list[tuple[str, ...]]:
    """Call order per block: one of :func:`orders`, then its reverse.

    Every order is used equally often, shuffled with a fixed seed.
    """
    rows = orders(arms, blocks)
    rows = rows * (blocks // len(rows))
    random.Random(SHUFFLE_SEED).shuffle(rows)
    return [row + row[::-1] for row in rows]


def summarize(samples: list[dict], arms: tuple[str, ...]) -> dict:
    """Per-arm and paired statistics recomputed from every raw sample."""
    seconds = {
        arm: [s["seconds"] for s in samples if s["arm"] == arm] for arm in arms
    }
    block_means: dict[tuple[int, str], float] = {}
    for block in sorted({s["block"] for s in samples}):
        for arm in arms:
            block_means[block, arm] = statistics.mean(
                s["seconds"]
                for s in samples
                if s["block"] == block and s["arm"] == arm
            )
    blocks = sorted({block for block, _ in block_means})

    result: dict = {"arms": {}, "pairs": {}}
    for arm, values in seconds.items():
        result["arms"][arm] = {
            "samples": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    for base, candidate in pairs(arms):
        base_mean = result["arms"][base]["mean"]
        effects = [
            block_means[block, base] - block_means[block, candidate]
            for block in blocks
        ]
        result["pairs"][f"{base}_to_{candidate}"] = {
            "reduction_pct": 100
            * (base_mean - result["arms"][candidate]["mean"])
            / base_mean,
            "block_effects": effects,
            "median_block_effect": statistics.median(effects),
            "positive_blocks": sum(effect > 0 for effect in effects),
            "blocks": len(effects),
        }
    return result
