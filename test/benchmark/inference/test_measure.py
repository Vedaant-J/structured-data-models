# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path

import pytest
import torch

from benchmark.inference.config import Config
from benchmark.inference.kernels import registry
from benchmark.inference.kernels.registry import Kernel
from benchmark.inference.report import summary, write
from benchmark.inference.runner import measure, verify_icl_attention
from benchmark.inference.schedule import is_optimized

SECONDS = {"eager": 3.0, "compiled": 2.0, "optimized": 1.5}


class FakeBench:
    """Fixed time per arm; the optimized arm selects and launches a kernel."""

    def __init__(
        self,
        optimized_probability: float = 0.25,
        eager_probability: float = 0.25,
        unavailable: str | None = None,
        compiled_kernels: tuple[str, ...] = ("triton_mm",),
        optimized_kernels: tuple[str, ...] = ("fake_gemm_gelu",),
    ) -> None:
        self.optimized_probability = optimized_probability
        self.eager_probability = eager_probability
        self.unavailable = unavailable
        self.kernels = {
            "eager": ("triton_mm",),
            "compiled": compiled_kernels,
            "optimized": optimized_kernels,
        }

    def _call(self, arm: str) -> None:
        if is_optimized(arm):
            args = torch.randn(2, 3), torch.randn(4, 3), None
            with registry.enabled():
                kernel = registry.select("linear_gelu", *args)
                if kernel is not None:
                    registry.launch(kernel, *args)

    def seconds(self, arm: str) -> float:
        self._call(arm)
        return SECONDS[arm.split(":")[0]]

    def snapshot(self, arm: str) -> dict[str, torch.Tensor]:
        self._call(arm)
        p = {
            "optimized": self.optimized_probability,
            "eager": self.eager_probability,
        }.get(arm.split(":")[0], 0.25)
        return {"probabilities": torch.tensor([[p, 1 - p]])}

    def profile(self, arm: str) -> dict[str, dict]:
        self._call(arm)
        base = arm.split(":")[0]
        return {
            name: {"count": 1, "ms": SECONDS[base]}
            for name in self.kernels[base]
        }


@pytest.fixture(autouse=True)
def _fake_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "KERNELS", {})
    registry.register(
        Kernel(
            op="linear_gelu",
            name="fake",
            supports=lambda x, weight, bias: None,
            impl=lambda x, weight, bias: x @ weight.T,
            cuda_symbols=("fake_gemm",),
        )
    )


def _config(tmp_path: Path, **kwargs: object) -> Config:
    kwargs = {"blocks": 6, "warmup": 2, **kwargs}
    return Config(
        model_size="small",
        context_rows=5000,
        output=tmp_path / "out",
        **kwargs,
    )


def test_completed_run_writes_all_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = measure(config, FakeBench())
    write(config.output, report)

    assert report["status"] == "completed"
    assert report["qualification"]["optimized_vs_compiled"]["bitwise_equal"]
    pairs = report["statistics"]["pairs"]
    assert pairs["compiled_to_optimized"]["reduction_pct"] == 25.0
    assert pairs["eager_to_optimized"]["positive_blocks"] == 6
    assert report["kernels"]["kernels_verified"]
    assert report["kernels"]["launches_per_optimized_call"] == {"fake": 1}

    with (config.output / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 * 2 * 3
    assert {float(row["seconds"]) for row in rows} == set(SECONDS.values())
    saved = json.loads((config.output / "report.json").read_text())
    assert saved["config"]["context_rows"] == 5000
    summary = (config.output / "summary.md").read_text()
    assert "25.00%" in summary
    assert "WARNING" not in summary
    with pytest.raises(FileExistsError):
        write(config.output, report)


def test_mismatch_blocks_timing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = measure(config, FakeBench(optimized_probability=0.5))
    write(config.output, report)

    assert report["status"] == "qualification_failed"
    gate = report["qualification"]["optimized_vs_compiled"]
    assert gate["first_mismatch"]["path"] == "probabilities"
    assert "statistics" not in report
    summary = (config.output / "summary.md").read_text()
    assert "NOT bitwise equal" in summary
    assert "Not timed" in summary


def test_mismatch_is_timed_when_allowed(tmp_path: Path) -> None:
    config = _config(tmp_path, allow_mismatch=True)
    report = measure(config, FakeBench(optimized_probability=0.5))
    assert report["status"] == "completed_with_mismatch"
    assert len(report["samples"]) == 6 * 2 * 3


def test_optimized_without_selected_kernel_falls_back_to_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "KERNELS", {})
    registry.register(
        Kernel(
            op="linear_gelu",
            name="sm100_only",
            supports=lambda x, weight, bias: "requires sm_100",
            impl=lambda x, weight, bias: x,
            cuda_symbols=("sm100",),
        )
    )
    report = measure(_config(tmp_path), FakeBench())
    assert "requires sm_100" in report["optimized_unavailable"]
    assert report["status"] == "completed_without_optimized"
    assert set(report["statistics"]["arms"]) == {"eager", "compiled"}


def test_unavailable_kernels_fall_back_to_native(tmp_path: Path) -> None:
    report = measure(_config(tmp_path), FakeBench(unavailable="no nvcc"))
    assert report["optimized_unavailable"] == "no nvcc"
    assert report["status"] == "completed_without_optimized"
    assert set(report["statistics"]["arms"]) == {"eager", "compiled"}


def test_eager_and_compiled_only(tmp_path: Path) -> None:
    config = _config(tmp_path, arms=("eager", "compiled"), blocks=2)
    report = measure(config, FakeBench())
    assert report["status"] == "completed"
    assert set(report["statistics"]["pairs"]) == {"eager_to_compiled"}
    assert (
        report["qualification"]["compiled_vs_eager"]["argmax_agreement_pct"]
        == 100.0
    )


def test_kernel_also_in_compiled_profile_is_not_verified(
    tmp_path: Path,
) -> None:
    bench = FakeBench(compiled_kernels=("triton_mm", "fake_gemm_gelu"))
    report = measure(_config(tmp_path), bench)
    assert report["status"] == "completed"
    assert not report["kernels"]["verification"]["fake"]["verified"]
    assert not report["kernels"]["kernels_verified"]


def test_kernel_with_a_missing_symbol_is_not_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "KERNELS", {})
    registry.register(
        Kernel(
            op="linear_gelu",
            name="two_launches",
            supports=lambda x, weight, bias: None,
            impl=lambda x, weight, bias: x @ weight.T,
            cuda_symbols=("fake_gemm", "fake_epilogue"),
        )
    )
    report = measure(_config(tmp_path), FakeBench())
    verification = report["kernels"]["verification"]["two_launches"]
    assert verification["optimized"] == ["fake_gemm_gelu"]
    assert not verification["verified"]
    assert not report["kernels"]["kernels_verified"]


def _register_tolerance_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "KERNELS", {})
    registry.register(
        Kernel(
            op="linear_gelu",
            name="fake",
            supports=lambda x, weight, bias: None,
            impl=lambda x, weight, bias: x @ weight.T,
            cuda_symbols=("fake_gemm",),
            bitwise=lambda x, weight, bias: False,
        )
    )


def test_tolerance_gate_admits_a_difference_within_eager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_tolerance_kernel(monkeypatch)
    bench = FakeBench(optimized_probability=0.26, eager_probability=0.3)
    report = measure(_config(tmp_path), bench)
    assert report["qualification"]["gate"] == "tolerance"
    assert report["status"] == "completed"


def test_tolerance_gate_rejects_a_difference_beyond_eager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_tolerance_kernel(monkeypatch)
    bench = FakeBench(optimized_probability=0.4, eager_probability=0.3)
    report = measure(_config(tmp_path), bench)
    assert report["status"] == "qualification_failed"
    assert "Tolerance gate failed" in summary(report)


def test_tolerance_gate_snapshots_eager_without_timing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_tolerance_kernel(monkeypatch)
    config = _config(tmp_path, arms=("compiled", "optimized"), blocks=2)
    bench = FakeBench(optimized_probability=0.26, eager_probability=0.3)
    report = measure(config, bench)
    assert report["status"] == "completed"
    assert not report["qualification"]["eager_timed"]
    assert "compiled_vs_eager" in report["qualification"]
    assert set(report["statistics"]["arms"]) == {"compiled", "optimized"}
    assert set(report["peak_memory"]) <= {"compiled", "optimized"}
    assert "(untimed)" in summary(report)


def test_optimized_variants_are_qualified_and_timed_per_arm(
    tmp_path: Path,
) -> None:
    arms = ("compiled", "optimized:off", "optimized:int8", "eager")
    config = _config(tmp_path, arms=arms, blocks=4)
    report = measure(config, FakeBench())
    write(config.output, report)

    assert report["status"] == "completed"
    assert set(report["qualification"]["arms"]) == set(arms[1:3])
    pairs = report["statistics"]["pairs"]
    assert pairs["compiled_to_optimized:int8"]["reduction_pct"] == 25.0
    assert "eager_to_compiled" in pairs
    for arm in arms[1:3]:
        assert report["kernels"]["arms"][arm]["kernels_verified"]
    assert "launches_per_optimized_call" not in report["kernels"]
    assert len(report["samples"]) == 4 * 2 * 4
    assert "optimized:int8 vs compiled bitwise equal" in summary(report)


def test_icl_attention_launch_verification() -> None:
    ok = {
        "a": {"census": 23, "profiled": 23},
        "b": {"census": 0, "profiled": 0},
    }
    verify_icl_attention(ok, "a", 23)
    verify_icl_attention(
        {k: {"census": 0, "profiled": 0} for k in ok}, None, 23
    )
    with pytest.raises(RuntimeError):
        verify_icl_attention(ok, None, 23)
    with pytest.raises(RuntimeError):
        verify_icl_attention({"a": {"census": 23, "profiled": 22}}, "a", 23)
    shared = {**ok, "b": {"census": 0, "profiled": None}}
    verify_icl_attention(shared, "a", 23)


def test_lossy_kernel_is_opt_in_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "KERNELS", {})
    registry.register(
        Kernel(
            op="linear_gelu",
            name="fake",
            supports=lambda x, weight, bias: None,
            impl=lambda x, weight, bias: x @ weight.T,
            cuda_symbols=("fake_gemm",),
            exact=False,
        )
    )
    config = _config(tmp_path, attention="int8")
    report = measure(config, FakeBench(optimized_probability=0.4))
    write(config.output, report)

    qualification = report["qualification"]
    assert report["status"] == "completed"
    assert qualification["gate"] == "opt-in"
    assert qualification["passed"]
    summary = (config.output / "summary.md").read_text()
    assert "optimized arm is lossy" in summary
    assert "`--attention int8`" in summary
    assert "compiled vs eager" in summary
