# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speed sweep over model sizes and contexts, one benchmark run per cell.

    python -m benchmark.inference.sweep run OUT [--sizes ...] [--contexts ...]
    python -m benchmark.inference.sweep table OUT

``run`` skips cells that already have ``result.json``, so it resumes. A cell
that runs out of memory is retried once with expandable segments, then
recorded as OOM. ``table`` writes ``OUT/tables.md`` and ``OUT/table.csv``.
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BF16_MAX_ROWS = 65_535
# Arm -> table column; a to d are the main comparison.
ARMS = {
    "compiled": "a compiled",
    "optimized:off": "b exact",
    "optimized:sm120": "c +bf16 attn",
    "optimized:int8": "d +int8 attn",
    "optimized:int8+ends": "int8+ends",
    "optimized:sm120+uncapped": "bf16 uncapped",
    "eager": "eager",
}
SMI_FIELDS = (
    "timestamp,power.draw,power.limit,clocks.sm,clocks.mem,"
    "temperature.gpu,utilization.gpu,memory.used"
)
OOM_MARKERS = ("OutOfMemoryError", "CUDA out of memory")
EXPANDABLE = "expandable_segments:True"


def _arms(size: str, rows: int) -> list[str]:
    arms = ["compiled", "optimized:off"]
    if rows <= BF16_MAX_ROWS:
        arms.append("optimized:sm120")
    arms.append("optimized:int8")
    if size == "large" and rows in (50_000, 100_000):
        arms.append("optimized:int8+ends")
    if size == "large" and rows == 100_000:
        arms.append("optimized:sm120+uncapped")
    if rows in (5_000, 50_000):
        arms.append("eager")
    return arms


def cells(sizes: list[str], contexts: list[int], repeat: bool) -> list[dict]:
    out = []
    for size in sizes:
        for rows in contexts:
            name = f"{size}-{rows}"
            arms = _arms(size, rows)
            out.append(
                {"name": name, "size": size, "rows": rows, "arms": arms}
            )
        if size == "large" and repeat:
            out += [
                {**c, "name": c["name"] + "-repeat"}
                for c in out
                if c["size"] == "large" and c["rows"] in (50_000, 100_000)
            ]
    return out


def blocks_for(arms: list[str], blocks: int) -> int:
    """The smallest multiple of the arm count not below ``blocks``."""
    return math.ceil(blocks / len(arms)) * len(arms)


def _attempt(
    cell: dict, args: argparse.Namespace, extra: list[str], expandable: bool
) -> dict:
    root = args.out / cell["name"]
    index = len(list(root.glob("attempt-*")))
    allocator = "expandable" if expandable else "default"
    run = root / f"attempt-{index:02d}-{allocator}"
    run.mkdir(parents=True)
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if expandable:
        env["PYTORCH_CUDA_ALLOC_CONF"] = EXPANDABLE
    command = [
        *(sys.executable, "-m", "benchmark.inference"),
        *("--model-size", cell["size"]),
        *("--context-rows", str(cell["rows"])),
        *("--arms", *cell["arms"]),
        *("--blocks", str(blocks_for(cell["arms"], args.blocks))),
        *("--warmup", str(args.warmup)),
        "--allow-mismatch",
        *("--output", str(run / "run")),
        *extra,
    ]
    smi_command = [
        *("nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,nounits"),
        *("-lms", "500", "-i", args.gpu),
    ]
    start = time.time()
    with (
        (run / "smi.csv").open("w") as smi_out,
        (run / "log.txt").open("w") as log,
    ):
        smi = subprocess.Popen(
            smi_command, stdout=smi_out, stderr=subprocess.DEVNULL
        )
        try:
            code = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=env
            ).returncode
        finally:
            smi.terminate()
            smi.wait()
    text = (run / "log.txt").read_text(errors="replace")
    report = run / "run" / "report.json"
    status = (
        json.loads(report.read_text())["status"] if report.exists() else None
    )
    return {
        "dir": run.name,
        "allocator": env.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
        "returncode": code,
        "seconds": time.time() - start,
        "status": status,
        "oom": any(marker in text for marker in OOM_MARKERS),
        "command": command[1:],
    }


def run(args: argparse.Namespace, extra: list[str]) -> None:
    plan = cells(args.sizes, args.contexts, not args.no_repeat)
    if args.only:
        plan = [c for c in plan if c["name"] in args.only]
    for cell in plan:
        arms = cell["arms"]
        blocks = blocks_for(arms, args.blocks)
        listed = " ".join(arms)
        print(f"{cell['name']}: {len(arms)} arms, {blocks} blocks: {listed}")
    if args.dry_run:
        return
    for cell in plan:
        result = args.out / cell["name"] / "result.json"
        if result.exists():
            print(f"skip {cell['name']} (done)", flush=True)
            continue
        print(f"run {cell['name']} ...", flush=True)
        attempts = [_attempt(cell, args, extra, expandable=False)]
        if attempts[-1]["status"] is None and attempts[-1]["oom"]:
            attempts.append(_attempt(cell, args, extra, expandable=True))
        last = attempts[-1]
        if last["status"]:
            outcome = "ok"
        else:
            outcome = "oom" if last["oom"] else "failed"
        record = {**cell, "outcome": outcome, "attempts": attempts}
        result.write_text(json.dumps(record, indent=2))
        line = (
            f"{cell['name']}: {outcome} ({last['status']}, "
            f"exit {last['returncode']}, {last['seconds']:.0f} s)"
        )
        print(line, flush=True)
        if outcome != "ok" or last["status"] != "completed":
            now = datetime.now().isoformat(timespec="seconds")
            with (args.out / "failures.log").open("a") as log:
                log.write(f"{now} {line} [{last['dir']}]\n")


def _smi(path: Path, window: list[float]) -> dict:
    """Median power, power limit and SM clock inside the timing window."""
    rows = []
    with path.open() as stream:
        for row in csv.reader(stream):
            try:
                at = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                power, limit, clock = (float(value) for value in row[1:4])
            except (ValueError, IndexError):
                continue
            if window[0] <= at.timestamp() <= window[1]:
                rows.append((power, limit, clock))
    if not rows:
        return {}
    power, limit, clock = zip(*rows)
    return {
        "power_w": statistics.median(power),
        "power_limit_w": statistics.median(limit),
        "sm_mhz": statistics.median(clock),
    }


def _load(out: Path) -> list[dict]:
    loaded = []
    for result in sorted(out.glob("*/result.json")):
        cell = json.loads(result.read_text())
        run = result.parent / cell["attempts"][-1]["dir"]
        report = run / "run" / "report.json"
        cell["report"] = (
            json.loads(report.read_text()) if report.exists() else None
        )
        window = (cell["report"] or {}).get("timing_window_unix")
        cell["smi"] = _smi(run / "smi.csv", window) if window else {}
        loaded.append(cell)
    return loaded


def _row(cell: dict, arm: str) -> dict:
    report = cell["report"]
    entry = {
        "cell": cell["name"],
        "size": cell["size"],
        "rows": cell["rows"],
        "arm": arm,
        "allocator": cell["attempts"][-1]["allocator"],
    }
    if report is None:
        return {**entry, "outcome": cell["outcome"]}
    if "statistics" not in report:
        return {**entry, "outcome": report["status"]}
    stats = report["statistics"]["arms"]
    if arm not in stats:
        return {**entry, "outcome": "not timed"}
    median = stats[arm]["median"]
    base = stats.get("compiled", {}).get("median")
    peak = report.get("peak_memory", {}).get(arm, {})
    gate = report["qualification"].get("arms", {}).get(arm, {})
    distance = gate.get("optimized_vs_compiled_distance", {})
    return {
        **entry,
        "outcome": "ok" if gate.get("passed", True) else "gate failed",
        "gate": gate.get("gate", ""),
        "median_ms": 1000 * median,
        "pct_vs_compiled": None
        if base is None
        else 100 * (median - base) / base,
        "speedup_vs_compiled": None if base is None else base / median,
        "peak_alloc_gib": peak.get("allocated_bytes", 0) / 2**30,
        "peak_reserved_gib": peak.get("reserved_bytes", 0) / 2**30,
        "max_dp_vs_compiled": distance.get("probability_max_abs_diff"),
        **cell["smi"],
    }


def _table_cell(cell: dict, arm: str) -> str:
    if arm not in cell["arms"]:
        capped = arm == "optimized:sm120" and cell["rows"] > BF16_MAX_ROWS
        return "= b (cap)" if capped else ""
    row = _row(cell, arm)
    if "median_ms" not in row:
        return row["outcome"].upper()
    pct = ""
    if row["pct_vs_compiled"] is not None and arm != "compiled":
        pct = f" ({row['pct_vs_compiled']:+.1f}%)"
    flag = " GATE FAIL" if row["outcome"] == "gate failed" else ""
    return f"{row['median_ms']:.0f}{pct} {row['peak_alloc_gib']:.1f}G{flag}"


def table(out: Path) -> None:
    loaded = _load(out)
    rows = [_row(c, arm) for c in loaded for arm in ARMS if arm in c["arms"]]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "Cells: median ms, % vs compiled (negative is faster), "
        "peak allocated GiB.",
        "",
    ]
    for size in ("large", "medium", "small"):
        sized = [c for c in loaded if c["size"] == size]
        if not sized:
            continue
        header = ["context", *ARMS.values(), "power W", "SM MHz"]
        lines += [
            f"## {size}",
            "",
            "| " + " | ".join(header) + " |",
            "|---" * len(header) + "|",
        ]
        for cell in sorted(sized, key=lambda c: (c["rows"], c["name"])):
            label = f"{cell['rows'] // 1000}k"
            if cell["name"].endswith("-repeat"):
                label += " (repeat)"
            if cell["attempts"][-1]["allocator"] != "default":
                label += " (expandable)"
            smi = cell["smi"]
            parts = [
                label,
                *(_table_cell(cell, arm) for arm in ARMS),
                f"{smi.get('power_w', 0):.0f}",
                f"{smi.get('sm_mhz', 0):.0f}",
            ]
            lines.append("| " + " | ".join(parts) + " |")
        lines.append("")
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    extra = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.inference.sweep"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run",
        help="run missing cells; arguments after -- go to every benchmark run",
    )
    run_parser.add_argument("out", type=Path)
    run_parser.add_argument(
        "--sizes", nargs="+", default=["large", "medium", "small"]
    )
    run_parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[5_000, 25_000, 50_000, 75_000, 100_000],
    )
    run_parser.add_argument(
        "--only", nargs="+", help="cell names, e.g. large-50000"
    )
    run_parser.add_argument(
        "--blocks",
        type=int,
        default=12,
        help="rounded up to a multiple of each cell's arm count",
    )
    run_parser.add_argument("--warmup", type=int, default=3)
    run_parser.add_argument("--no-repeat", action="store_true")
    run_parser.add_argument(
        "--gpu",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
    )
    run_parser.add_argument("--dry-run", action="store_true")
    commands.add_parser("table").add_argument("out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        args.out.mkdir(parents=True, exist_ok=True)
        run(args, extra)
    table(args.out)


if __name__ == "__main__":
    main()
