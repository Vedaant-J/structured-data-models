# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_covtype

import sdm
from benchmark.inference.config import Config

Tables = tuple[sdm.TableTensor, sdm.TableTensor, sdm.TableTensor]


def _rows(config: Config, available: int) -> np.ndarray:
    rows = config.context_rows + config.query_rows
    if rows > available:
        raise SystemExit(
            f"{config.dataset} has {available} rows; --context-rows "
            f"{config.context_rows} plus --query-rows {config.query_rows} "
            "do not fit"
        )
    return np.random.default_rng(config.seed).permutation(available)[:rows]


def _covertype(config: Config) -> tuple[sdm.TableTensor, sdm.TableTensor]:
    dataset = fetch_covtype()
    rows = _rows(config, len(dataset.target))
    x = torch.from_numpy(dataset.data[rows]).float()  # [R, 54]
    names = list(dataset.feature_names)
    # 10 numerical columns, then 44 binary indicators as categoricals:
    features = sdm.TableTensor(
        columns={
            sdm.Stype.numerical: names[:10],
            sdm.Stype.categorical: names[10:],
        },
        numerical=x[:, :10].contiguous(),
        categorical=sdm.CategoricalTensor(
            code=x[:, 10:].long(),
            categories=tuple(torch.tensor([0, 1]) for _ in range(44)),
        ),
    )
    labels = torch.from_numpy(dataset.target[rows]).long()
    target = sdm.TableTensor(
        columns={sdm.Stype.categorical: ["cover_type"]},
        categorical=sdm.CategoricalTensor(
            code=(labels - 1).unsqueeze(-1),
            categories=(torch.arange(1, 8),),
        ),
    )
    return features, target


def _table(config: Config) -> tuple[sdm.TableTensor, sdm.TableTensor]:
    assert config.data is not None
    assert config.target is not None
    if config.data.suffix == ".parquet":
        df = pd.read_parquet(config.data)
    else:
        df = pd.read_csv(config.data)
    if config.target not in df.columns:
        raise SystemExit(f"--target {config.target!r} is not a column")
    df = df.iloc[_rows(config, len(df))].reset_index(drop=True)
    stypes = sdm.infer_stypes(df, overrides={config.target: "categorical"})
    table = sdm.TableTensor.from_pandas(df, stypes=stypes)
    return table.drop_columns(config.target), table[:, config.target]


def tables(config: Config) -> Tables:
    """Context features, context labels and query features on the device.

    Rows are a fixed-seed permutation: context is its first rows, query
    the rows right after.
    """
    features, target = (
        _covertype(config) if config.data is None else _table(config)
    )
    context, device = config.context_rows, config.device
    return (
        features[:context].to(device),
        target[:context].to(device),
        features[context:].to(device),
    )
