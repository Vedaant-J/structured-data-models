# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-to-run determinism of fresh fit + predict, per arm.

    python -m benchmark.attention.determinism --out FILE [--size large] \
        [--rows 50000] [--arms compiled optimized] [--runs 10] \
        [--attention sm120]

Repeats fit + predict on identical inputs and compares every snapshot (the
fitted cache and the probabilities) bitwise with the arm's first run. The
optimized arm runs only the ``icl_attention`` op, with ``--attention``
(default ``$SDM_ICL_ATTENTION``, else ``sm120``). The kernel-level check is
``test_split_merge_is_deterministic`` in ``test/benchmark/inference``.
"""

import argparse
import json
import os
from pathlib import Path

from benchmark.inference.config import Config
from benchmark.inference.kernels import registry
from benchmark.inference.runner import CudaBench
from benchmark.inference.snapshot import compare

WARMUP_CALLS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", default="large")
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--arms", nargs="+", default=["compiled", "optimized"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--attention", default=os.environ.get(registry.ATTENTION_ENV, "sm120")
    )
    args = parser.parse_args()
    config = Config(
        model_size=args.size,
        context_rows=args.rows,
        output=args.out,
        arms=tuple(args.arms),
        ops=("icl_attention",),
        attention=args.attention,
    )
    bench = CudaBench(config)
    report = {}
    for arm in args.arms:
        for _ in range(WARMUP_CALLS):
            bench.seconds(arm)
        first = bench.snapshot(arm)
        differing = [
            run
            for run in range(1, args.runs)
            if not compare(first, bench.snapshot(arm))["bitwise_equal"]
        ]
        report[arm] = {
            "runs": args.runs,
            "leaves": len(first),
            "differing_runs": differing,
        }
        counts = f"{len(differing)}/{args.runs - 1}"
        print(f"{arm:>10} {counts} runs differ", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
