# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark.inference.kernels import registry
from benchmark.inference.schedule import is_optimized, orders

ARMS = ("eager", "compiled", "optimized")


@dataclass(frozen=True)
class Config:
    model_size: str
    context_rows: int
    output: Path
    checkpoint: Path | None = None
    kernels: Path | None = None
    ops: tuple[str, ...] | None = None
    attention: str = "sm120"
    data: Path | None = None
    target: str | None = None
    query_rows: int = 16
    device: str = "cuda:0"
    arms: tuple[str, ...] = ARMS
    blocks: int = 12
    warmup: int = 3
    precision: str = "ieee"
    seed: int = 0
    allow_mismatch: bool = False
    qualify_only: bool = False
    shadow: bool = False

    def arm_attention(self, arm: str) -> str:
        return arm_attention(arm, self.attention)

    @property
    def dataset(self) -> str:
        return "covertype" if self.data is None else self.data.name

    def record(self) -> dict:
        record = asdict(self)
        # Only file names: local directory layouts are not part of a result.
        for key in ("output", "checkpoint", "kernels", "data"):
            path = record[key]
            record[key] = None if path is None else path.name
        record["dataset"] = self.dataset
        return record


def arm_attention(arm: str, default: str) -> str:
    """ICL attention setting of ``optimized`` or ``optimized:<attention>``."""
    if arm == "optimized":
        return default
    _, setting, *rest = arm.split(":")
    if rest:
        raise ValueError(f"arm {arm!r}: expected optimized:<attention>")
    registry.parse_attention(setting)
    return setting


def _arm(text: str) -> str:
    if text in ARMS:
        return text
    if not is_optimized(text):
        raise argparse.ArgumentTypeError(
            f"arm must be one of {ARMS} or optimized:<attention>, got {text}"
        )
    try:
        arm_attention(text, "sm120")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return text


def _positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {text}"
        )
    return value


def parse(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.inference",
        description="Latency of fresh-context KumoTabular fit + predict.",
    )
    parser.add_argument(
        "--model-size",
        required=True,
        choices=("small", "medium", "large"),
    )
    parser.add_argument("--context-rows", required=True, type=_positive)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="classifier state dict; default downloads the pretrained one",
    )
    parser.add_argument(
        "--kernels",
        type=Path,
        help="kernel module directory for the optimized arms; missing "
        "modules are built into it (default: a cache directory)",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        help="limit the optimized arm to these named ops (default: all)",
    )
    parser.add_argument(
        "--attention",
        default=os.environ.get(registry.ATTENTION_ENV, "sm120"),
        help="ICL attention kernel of the optimized arm: sm120 (BF16), fa2, "
        "off (native, bitwise), or the lossy opt-in int8[+ends] "
        "(default: $SDM_ICL_ATTENTION, else sm120)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help=".parquet or .csv table; default is Covertype",
    )
    parser.add_argument("--target", help="target column of --data")
    parser.add_argument("--query-rows", type=_positive, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--arms",
        nargs="+",
        type=_arm,
        default=ARMS,
        help="eager, compiled, optimized, and optimized:<attention> arms, "
        "each compiled with its own ICL attention setting",
    )
    parser.add_argument("--blocks", type=_positive, default=12)
    parser.add_argument(
        "--warmup",
        type=_positive,
        default=3,
        help="untimed calls per arm; the first includes compilation",
    )
    parser.add_argument(
        "--precision",
        choices=("ieee", "default"),
        default="ieee",
        help="ieee disables TF32 and reduced-precision BF16 reductions",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="time optimized arms even if they fail their gate",
    )
    parser.add_argument(
        "--qualify-only",
        action="store_true",
        help="warm up and qualify the arms, then stop before timing",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="run each selected kernel beside native on native's inputs and "
        "report per op whether its outputs are bitwise equal; no timing",
    )
    args = parser.parse_args(argv)
    args.arms = tuple(args.arms)
    args.ops = None if args.ops is None else tuple(args.ops)

    if (args.data is None) != (args.target is None):
        parser.error("--data and --target must be given together")
    if args.data is not None and args.data.suffix not in (".parquet", ".csv"):
        parser.error("--data must be a .parquet or .csv file")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint {args.checkpoint} does not exist")
    try:
        registry.parse_attention(args.attention)
    except ValueError as error:
        parser.error(f"--attention: {error}")
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms must not repeat an arm")
    if (
        any(is_optimized(arm) for arm in args.arms)
        and "compiled" not in args.arms
    ):
        parser.error(
            "--arms: an optimized arm requires compiled, which it is "
            "qualified against"
        )
    try:
        orders(args.arms, args.blocks)
    except ValueError as error:
        parser.error(
            f"--blocks: {error}, so every arm appears equally often in every "
            "position"
        )
    if args.output.exists():
        parser.error(f"output {args.output} already exists; pick a new one")
    return Config(**vars(args))
