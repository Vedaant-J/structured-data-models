# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest

from benchmark.inference.kernels import build

MODULE = "linear_gelu_sm120"


def test_stamp_tracks_module_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(build.CSRC / MODULE, tmp_path / MODULE)
    monkeypatch.setattr(build, "CSRC", tmp_path)
    first = build.stamp(MODULE)
    assert first == build.stamp(MODULE)
    source = next((tmp_path / MODULE).iterdir())
    source.write_text(source.read_text() + "\n")
    assert build.stamp(MODULE) != first


def test_up_to_date_module_skips_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / f"{MODULE}.so").write_bytes(b"")
    (tmp_path / f"{MODULE}.sha256").write_text(build.stamp(MODULE))
    monkeypatch.setattr(
        build, "check_toolchain", lambda: pytest.fail("rebuilt")
    )
    assert build.build(tmp_path, [MODULE]) == tmp_path


def test_stale_module_needs_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / f"{MODULE}.so").write_bytes(b"")
    (tmp_path / f"{MODULE}.sha256").write_text("stale")
    monkeypatch.setattr(build.cpp_extension, "CUDA_HOME", None)
    with pytest.raises(RuntimeError, match="CUDA toolkit"):
        build.build(tmp_path, [MODULE])
