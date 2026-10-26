# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the ICL attention accuracy-suite tables as parquet files.

    python -m benchmark.attention.datasets DIR [--names adult diamonds ...]

Classification tables keep their OpenML columns, with the target last as a
string column ``class``; Covertype is the benchmark's built-in dataset and is
not written. Regression tables are TabArena v0.1 regression tasks with a
numeric ``target`` column. ``DIR/manifest.json`` records each table's task,
file, target column, rows, features and classes. OpenML data is cached under
``$SCIKIT_LEARN_DATA``.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

CLASSIFICATION = {
    "adult": 1590,
    "letter": 6,
    "electricity": 151,
    "magic_telescope": 1120,
}
REGRESSION = {
    "diamonds": (46923, "price"),
    "physiochemical_protein": (46949, "ResidualSize"),
    "superconductivity": (46961, "critical_temp"),
    "houses": (46934, "LnMedianHouseValue"),
    "food_delivery_time": (46928, "Time_taken(min)"),
    "qsar_tid_11": (46953, "MEDIAN_PXC50"),
}


def categoricals_as_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with categorical columns as objects, missing as None."""
    df = df.copy()
    for column in df.columns:
        if isinstance(df[column].dtype, pd.CategoricalDtype):
            values = df[column].astype(object)
            df[column] = values.where(df[column].notna(), None)
    return df


def _classification(name: str) -> pd.DataFrame:
    frame = fetch_openml(data_id=CLASSIFICATION[name], as_frame=True).frame
    frame = frame.rename(columns={frame.columns[-1]: "class"})
    frame["class"] = frame["class"].astype(str)
    return frame


def regression_table(name: str) -> pd.DataFrame:
    """Features of a ``REGRESSION`` table and its response as ``target``."""
    data_id, column = REGRESSION[name]
    data = fetch_openml(
        data_id=data_id, as_frame=True, parser="auto", target_column=column
    )
    df = categoricals_as_objects(data.data)
    df["target"] = data.target.astype(float).to_numpy()
    return df


def fetch(name: str, directory: Path) -> dict:
    regression = name in REGRESSION
    if regression:
        df, target = regression_table(name), "target"
    else:
        df, target = _classification(name), "class"
    path = directory / f"{name}.parquet"
    df.to_parquet(path, index=not regression)
    return {
        "task": "regression" if regression else "classification",
        "file": path.name,
        "target": target,
        "rows": len(df),
        "features": df.shape[1] - 1,
        "classes": None if regression else int(df[target].nunique()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--names", nargs="+", default=[*CLASSIFICATION, *REGRESSION]
    )
    args = parser.parse_args(argv)
    args.directory.mkdir(parents=True, exist_ok=True)
    path = args.directory / "manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {}
    for name in args.names:
        manifest[name] = fetch(name, args.directory)
        print(name, manifest[name], flush=True)
        path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
