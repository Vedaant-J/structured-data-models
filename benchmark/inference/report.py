# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

import sdm
from benchmark.inference.config import arm_attention


def _run(*command: str) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def revision(path: Path) -> dict:
    """Git revision of ``path``'s checkout, else a hash of its Python files."""
    head = _run("git", "-C", str(path), "rev-parse", "HEAD")
    if head is None:
        digest = hashlib.sha256()
        for file in sorted(path.rglob("*.py")):
            digest.update(file.relative_to(path).as_posix().encode())
            digest.update(file.read_bytes())
        return {"content_sha256": digest.hexdigest()}
    status = _run("git", "-C", str(path), "status", "--porcelain", "--", ".")
    return {"revision": head, "dirty": bool(status)}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment(device: torch.device) -> dict:
    properties = torch.cuda.get_device_properties(device)
    driver = _run(
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
    )
    return {
        "sdm": revision(Path(sdm.__file__).parent),
        "benchmark": revision(Path(__file__).parent),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": _version("triton"),
        "gpu": {
            "name": properties.name,
            "capability": f"{properties.major}.{properties.minor}",
            "memory_gib": round(properties.total_memory / 2**30, 1),
            "driver": driver.splitlines()[0] if driver else "unknown",
        },
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }


def _ms(seconds: float) -> str:
    return f"{1000 * seconds:.2f}"


def _gib(value: int) -> str:
    return f"{value / 2**30:.2f}"


def _qualification(
    arm: str, setting: str, entry: dict, scale: dict | None
) -> list[str]:
    distance = entry["optimized_vs_compiled_distance"]
    gate = entry["optimized_vs_compiled"]
    if entry["gate"] == "opt-in":
        lines = [
            f"**The {arm} arm is lossy: opt-in 8-bit ICL attention "
            f"(`--attention {setting}`).** It is not bitwise equal to "
            "compiled and is timed without a tolerance gate. Distance from "
            f"compiled: max |dp| {distance['probability_max_abs_diff']:.3g}, "
            f"mean |dp| {distance['probability_mean_abs_diff']:.3g}, argmax "
            f"agreement {distance['argmax_agreement_pct']:.2f}%."
        ]
        if scale is not None:
            lines.append(
                "For scale, compiled vs eager: max |dp| "
                f"{scale['probability_max_abs_diff']:.3g}, mean |dp| "
                f"{scale['probability_mean_abs_diff']:.3g}, argmax agreement "
                f"{scale['argmax_agreement_pct']:.2f}%."
            )
        return lines
    if entry["gate"] == "tolerance":
        verdict = "passed" if entry["passed"] else "failed"
        return [
            f"Tolerance gate {verdict} ({arm}): {arm} vs compiled max |dp| "
            f"{distance['probability_max_abs_diff']:.3g} (allowed "
            f"{scale['probability_max_abs_diff']:.3g}, the compiled vs eager "
            f"distance), argmax agreement "
            f"{distance['argmax_agreement_pct']:.2f}%."
        ]
    if not gate["bitwise_equal"]:
        return [
            f"**WARNING: {arm} output is NOT bitwise equal to compiled** "
            f"({gate['mismatched_leaves']} of {gate['leaves']} leaves "
            f"differ; first: `{json.dumps(gate['first_mismatch'])}`)."
        ]
    return [f"{arm} vs compiled bitwise equal: True."]


def summary(report: dict) -> str:
    config, env = report["config"], report.get("environment", {})
    gpu = env.get("gpu", {})
    lines = [
        "# KumoTabular fit + predict latency",
        "",
        f"Setup: {config['model_size']} model, {config['context_rows']} "
        f"context rows, {config['query_rows']} query rows, "
        f"{config['dataset']}, {gpu.get('name', '?')} "
        f"(sm {gpu.get('capability', '?')}), torch {env.get('torch', '?')}, "
        f"precision {config['precision']}, {config['blocks']} blocks.",
        "",
        f"Status: **{report['status']}**",
        "",
    ]
    if "optimized_unavailable" in report:
        lines += [
            "**Optimized arm skipped** (native arms only): "
            f"{report['optimized_unavailable']}.",
            "",
        ]
    if report["status"] in ("shadow_bitwise", "shadow_mismatch"):
        lines += [
            "Shadow check (kernel vs native output on native's inputs):",
            "",
            "| arm | op | outputs | bitwise | max abs diff |",
            "|---|---|---|---|---|",
        ]
        for arm, entry in report["kernels"]["arms"].items():
            for op, e in sorted(entry["shadow"].items()):
                lines.append(
                    f"| {arm} | {op} | {e['outputs']} | {e['bitwise']} | "
                    f"{e['max_abs_diff']:.3g} |"
                )
        return "\n".join(lines) + "\n"
    qualification = report["qualification"]
    scale = qualification.get("compiled_vs_eager")
    for arm, entry in qualification.get("arms", {}).items():
        setting = arm_attention(arm, config["attention"])
        lines += [*_qualification(arm, setting, entry, scale), ""]
    if scale is not None:
        timed = "" if qualification.get("eager_timed", True) else " (untimed)"
        lines += [
            f"Compiled vs eager{timed}: max |dp| "
            f"{scale['probability_max_abs_diff']:.3g}, argmax agreement "
            f"{scale['argmax_agreement_pct']:.2f}%.",
            "",
        ]
    if "statistics" not in report:
        lines.append(
            "Not timed: --qualify-only."
            if report["status"] == "qualified"
            else "Not timed: qualification failed."
        )
        return "\n".join(lines) + "\n"

    stats, peak = report["statistics"], report.get("peak_memory", {})
    lines += [
        "| arm | mean ms | median ms | min ms | stdev ms | samples | "
        "peak alloc GiB | peak reserved GiB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, s in stats["arms"].items():
        memory = peak.get(arm)
        lines.append(
            f"| {arm} | {_ms(s['mean'])} | {_ms(s['median'])} | "
            f"{_ms(s['min'])} | {_ms(s['stdev'])} | {s['samples']} | "
            + (
                f"{_gib(memory['allocated_bytes'])} | "
                f"{_gib(memory['reserved_bytes'])} |"
                if memory
                else "- | - |"
            )
        )
    lines.append("")
    for name, pair in stats["pairs"].items():
        lines.append(
            f"- {name.replace('_to_', ' -> ')}: {pair['reduction_pct']:.2f}% "
            f"lower mean latency; {pair['positive_blocks']}/{pair['blocks']} "
            f"blocks faster; median block effect "
            f"{_ms(pair['median_block_effect'])} ms"
        )
    lines.append("")
    for arm, entry in report["kernels"].get("arms", {}).items():
        if entry["verification"]:
            lines.append(
                f"Kernels verified in the {arm} profile: "
                f"{entry['kernels_verified']}."
            )
            for name, check in entry["verification"].items():
                verdict = "ran" if check["verified"] else "NOT verified"
                lines.append(f"- {name}: {verdict}")
    return "\n".join(lines) + "\n"


def write(output: Path, report: dict) -> None:
    """Write report.json, samples.csv and summary.md to a new directory."""
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("block", "position", "arm", "seconds")
        )
        writer.writeheader()
        writer.writerows(report.get("samples", []))
    (output / "summary.md").write_text(summary(report))
