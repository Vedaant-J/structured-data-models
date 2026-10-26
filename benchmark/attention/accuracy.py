# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classification accuracy of the ICL attention kernel against compiled SDM.

    python -m benchmark.attention.accuracy --data-dir DIR --out DIR \
        [--context-rows 10000] [--sizes large medium] [--query-rows 16 1024] \
        [--seed 0] [--datasets covertype adult ...]

Runs ``python -m benchmark.inference --qualify-only`` (arms eager, compiled
and optimized with every kernel; no timing) per combination. The other
kernels are bitwise, so optimized differs from compiled only through ICL
attention. The environment is passed through, so ``SDM_ICL_ATTENTION=fa2``
measures FlashAttention 2 instead. ``--data-dir`` is where
``benchmark.attention.datasets`` wrote its files. Writes ``accuracy.md`` and
``accuracy.json`` to ``--out``.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

COVERTYPE = {
    "dataset": "covertype",
    "rows": 581012,
    "features": 54,
    "classes": 7,
}


def catalog(directory: Path, names: list[str] | None) -> list[dict]:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {}
    entries = [{**COVERTYPE, "file": None, "target": None}] + [
        {"dataset": name, **info}
        for name, info in manifest.items()
        if info["task"] == "classification"
    ]
    return [e for e in entries if names is None or e["dataset"] in names]


def run(
    entry: dict, size: str, queries: int, args: argparse.Namespace
) -> dict:
    out = (
        args.out
        / "runs"
        / f"{entry['dataset']}-{size}-q{queries}-s{args.seed}"
    )
    command = [
        *(sys.executable, "-m", "benchmark.inference"),
        *("--model-size", size),
        *("--context-rows", str(args.context_rows)),
        *("--query-rows", str(queries)),
        *("--arms", "eager", "compiled", "optimized"),
        "--qualify-only",
        *("--seed", str(args.seed)),
        *("--output", str(out)),
    ]
    if entry["file"] is not None:
        command += ["--data", str(args.data_dir / entry["file"])]
        command += ["--target", entry["target"]]
    subprocess.run(command, check=False)
    row = {
        "dataset": entry["dataset"],
        "rows": entry["rows"],
        "features": entry["features"],
        "classes": entry["classes"],
        "size": size,
        "seed": args.seed,
        "query_rows": queries,
        "pass": False,
    }
    report_path = out / "report.json"
    if not report_path.is_file():
        return row
    report = json.loads(report_path.read_text())
    qualification = report["qualification"]
    chosen = {d["kernel"] for d in report["kernels"]["decisions"]}
    row["kernels"] = sorted(chosen - {"native"})
    if "optimized_vs_compiled_distance" in qualification:
        compiled = qualification["compiled_vs_eager"]
        optimized = qualification["optimized_vs_compiled_distance"]
        row.update(
            compiled_vs_eager_pmax=compiled["probability_max_abs_diff"],
            compiled_vs_eager_agree=compiled["argmax_agreement_pct"],
            optimized_vs_compiled_pmax=optimized["probability_max_abs_diff"],
            optimized_vs_compiled_agree=optimized["argmax_agreement_pct"],
        )
    row["status"] = report["status"]
    row["launches"] = report["kernels"].get("icl_attention_launches")
    row["pass"] = bool(
        qualification["passed"]
        and row["kernels"]
        and report["status"] == "qualified"
    )
    return row


COLUMNS = (
    ("dataset", "dataset"),
    ("rows", "rows"),
    ("features", "features"),
    ("classes", "classes"),
    ("size", "size"),
    ("seed", "seed"),
    ("query_rows", "query rows"),
    ("compiled_vs_eager_pmax", "compiled vs eager max |dp|"),
    ("compiled_vs_eager_agree", "compiled vs eager argmax agree %"),
    ("optimized_vs_compiled_pmax", "optimized vs compiled max |dp|"),
    ("optimized_vs_compiled_agree", "optimized vs compiled argmax agree %"),
    ("pass", "pass"),
)


def markdown(rows: list[dict]) -> str:
    def cell(value: object) -> str:
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    lines = [
        "| " + " | ".join(h for _, h in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(cell(row.get(k, "")) for k, _ in COLUMNS) + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-rows", type=int, default=10000)
    parser.add_argument("--sizes", nargs="+", default=["large"])
    parser.add_argument(
        "--query-rows", nargs="+", type=int, default=[16, 1024]
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        run(entry, size, queries, args)
        for entry in catalog(args.data_dir, args.datasets)
        for size in args.sizes
        for queries in args.query_rows
    ]
    (args.out / "accuracy.json").write_text(json.dumps(rows, indent=2))
    text = markdown(rows)
    (args.out / "accuracy.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
