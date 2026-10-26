# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the tables of ``benchmark.attention.compare`` as parquet files.

    python -m benchmark.attention.compare_data --out DIR [--names adult]

Each table gets a ``target`` column: integer class codes, or the numeric
response for the regression tables, which are those of
``benchmark.attention.datasets``. ``manifest.json`` records rows, features
and classes.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml

from benchmark.attention.datasets import (
    REGRESSION,
    categoricals_as_objects,
    regression_table,
)

TARGET = "target"
CLASSIFICATION = {
    "adult": 1590,
    "higgs": 23512,
    "volkert": 41166,
    "letter": 6,
    "bank_marketing": 1461,
}


def _covertype() -> pd.DataFrame:
    data = fetch_covtype(as_frame=True)
    df = data.data.copy()
    for column in df.columns[10:]:
        df[column] = df[column].astype(int).astype(str)
    df[TARGET] = data.target.to_numpy().astype(int) - 1
    return df


def _classification(name: str) -> pd.DataFrame:
    data = fetch_openml(
        data_id=CLASSIFICATION[name],
        as_frame=True,
        parser="auto",
        target_column="default-target",
    )
    df = categoricals_as_objects(data.data)
    df[TARGET] = pd.factorize(data.target.astype(str), sort=True)[0]
    return df


def build(name: str) -> pd.DataFrame:
    if name == "covertype":
        return _covertype()
    if name in REGRESSION:
        return regression_table(name)
    return _classification(name)


def describe(name: str, df: pd.DataFrame) -> dict:
    features = df.drop(columns=TARGET)
    categorical = [
        c
        for c in features.columns
        if not pd.api.types.is_numeric_dtype(features[c])
    ]
    task = "regression" if name in REGRESSION else "classification"
    return {
        "rows": len(df),
        "features": features.shape[1],
        "categorical_features": len(categorical),
        "task": task,
        "classes": int(df[TARGET].nunique())
        if task == "classification"
        else None,
        "target": TARGET,
        "file": f"{name}.parquet",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--names",
        nargs="+",
        default=["covertype", *CLASSIFICATION, *REGRESSION],
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    for name in args.names:
        df = build(name)
        df.to_parquet(args.out / f"{name}.parquet", index=False)
        manifest[name] = describe(name, df)
        print(name, manifest[name], flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
