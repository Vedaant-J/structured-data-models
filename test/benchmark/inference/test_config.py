# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from benchmark.inference.config import ARMS, parse

_ARGS = ["--model-size", "large", "--context-rows", "50000"]


def test_defaults(tmp_path: Path) -> None:
    config = parse([*_ARGS, "--output", str(tmp_path / "out")])
    assert config.arms == ARMS
    assert config.query_rows == 16
    assert config.precision == "ieee"
    assert config.checkpoint is None
    assert config.record()["dataset"] == "covertype"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--model-size", "huge"], "invalid choice"),
        (["--context-rows", "0"], "positive integer"),
        (["--blocks", "4"], "multiple of 3"),
        (["--arms", "compiled", "optimized:fp8"], "ICL attention"),
        (["--arms", "compiled", "optimized:off:x"], "optimized:<attention>"),
        (["--arms", "eager", "optimized:off"], "requires compiled"),
        (["--data", "table.csv"], "--data and --target"),
        (["--data", "table.txt", "--target", "y"], ".parquet or .csv"),
        (["--arms", "eager", "optimized"], "requires compiled"),
        (["--arms", "eager", "eager"], "must not repeat"),
        (["--checkpoint", "missing.pt"], "does not exist"),
    ],
)
def test_invalid_configuration_exits_with_message(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    message: str,
) -> None:
    with pytest.raises(SystemExit):
        parse([*_ARGS, "--output", str(tmp_path / "out"), *args])
    assert message in capsys.readouterr().err


def test_existing_output_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        parse([*_ARGS, "--output", str(tmp_path)])
    assert "already exists" in capsys.readouterr().err


def test_attention_comes_from_the_flag_or_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SDM_ICL_ATTENTION", raising=False)
    assert (
        parse([*_ARGS, "--output", str(tmp_path / "a")]).attention == "sm120"
    )
    monkeypatch.setenv("SDM_ICL_ATTENTION", "off")
    assert parse([*_ARGS, "--output", str(tmp_path / "b")]).attention == "off"
    config = parse(
        [*_ARGS, "--output", str(tmp_path / "c"), "--attention", "int8+ends"]
    )
    assert config.record()["attention"] == "int8+ends"


def test_attention_is_validated(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse([*_ARGS, "--output", str(tmp_path / "o"), "--attention", "fp8"])


def test_optimized_variants(tmp_path: Path) -> None:
    config = parse(
        [
            *(*_ARGS, "--output", str(tmp_path / "out"), "--blocks", "4"),
            *("--arms", "compiled", "optimized", "optimized:int8+ends"),
            "optimized:off",
        ]
    )
    assert config.arm_attention("optimized") == "sm120"
    assert config.arm_attention("optimized:int8+ends") == "int8+ends"
    assert config.arm_attention("optimized:off") == "off"
