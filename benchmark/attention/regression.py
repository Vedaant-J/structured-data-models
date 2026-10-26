# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression accuracy of the ICL attention kernel against compiled SDM.

    python -m benchmark.attention.regression run --data-dir DIR --out RES \
        [--settings sm120] [--contexts 50000] [--queries 16 1024] \
        [--seeds 0 1 2]
    python -m benchmark.attention.regression table --results RES --out FILE.md

``run`` writes one JSON (and an npz of the 999 predicted quantiles) per
table, context, query rows and seed, with the eager and compiled arms and
compiled plus only the ``icl_attention`` op for each setting. Every arm is
checked with the op census and the CUDA profiler: eager and compiled launch
no ICL attention kernel, a setting arm launches its kernel once per fit call.
Differences are |delta| / context target std, over all quantiles and on their
mean (the point prediction that RMSE and R2 score).
"""

import argparse
import gc
import json
import os
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.profiler import ProfilerActivity, profile

import sdm
from benchmark.attention.compare import LEVELS, build_tables
from benchmark.inference import runner
from benchmark.inference.kernels import integration, registry

EXPECTED = {
    "sm120": "sm120_icl_attention_d64",
    "fa2": "fa2_icl_attention_d64",
}


class Arms:
    def __init__(self, model, settings, device):
        self.model, self.device = model, device
        self.expected = {"eager": None, "compiled": None}
        self.expected.update({s: EXPECTED[s] for s in settings})
        self.inner = model.models[sdm.Task.regression]
        self.calls = len(self.inner.icl_block.layers) - 1
        self.forwards = {
            "compiled": torch.compile(
                self.inner.forward, **integration.COMPILE
            )
        }
        for setting in settings:
            self.forwards[setting] = integration.compile_with_kernels(
                self.inner.forward, ops=("icl_attention",)
            )

    def predict(self, arm, tables, seed):
        x, y, query = tables
        os.environ[registry.ATTENTION_ENV] = arm if arm in EXPECTED else "off"
        self.model.clear()
        registry.reset_census()
        generator = torch.Generator(self.device).manual_seed(seed)
        with ExitStack() as stack:
            if arm != "eager":
                self.inner.forward = self.forwards[arm]
                stack.callback(vars(self.inner).pop, "forward")
            stack.enter_context(torch.inference_mode())
            stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
            prof = stack.enter_context(
                profile(activities=[ProfilerActivity.CUDA])
            )
            self.model.fit(x=x, y=y, num_estimators=1, generator=generator)
            result = self.model.predict(query)
            torch.cuda.synchronize(self.device)
        out = result.numerical.detach().float().cpu().numpy()
        self.model.clear()
        launches = runner.icl_attention_launches(
            registry.census()["launches"],
            runner.kernel_counts(prof),
            self.expected[arm],
        )
        runner.verify_icl_attention(launches, self.expected[arm], self.calls)
        return out, launches


def metrics(quantiles, truth):
    quantiles = np.sort(quantiles.astype(np.float64), axis=1)
    mean = quantiles.mean(1)
    err = truth[:, None] - quantiles
    pinball = np.maximum(LEVELS * err, (LEVELS - 1) * err)
    out = {
        "rmse": float(np.sqrt(((mean - truth) ** 2).mean())),
        "r2": float(r2_score(truth, mean)),
        "pinball": float(pinball.mean()),
    }
    for name, lo, hi in (("80", 99, 899), ("90", 49, 949)):
        inside = (truth >= quantiles[:, lo]) & (truth <= quantiles[:, hi])
        out[f"coverage_{name}"] = float(inside.mean())
    return out


def distance(reference, candidate, scale):
    ref, cand = reference.astype(np.float64), candidate.astype(np.float64)
    diff = np.abs(ref - cand) / scale
    point = np.abs(ref.mean(1) - cand.mean(1)) / scale
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p999_abs_diff": float(np.quantile(diff, 0.999)),
        "point_max": float(point.max()),
        "point_mean": float(point.mean()),
    }


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    names = args.datasets or [
        n for n, i in manifest.items() if i["task"] == "regression"
    ]
    integration.isolate_compiles()
    runner.set_precision("ieee")
    arms = ["eager", "compiled", *args.settings]
    for name in names:
        info = manifest[name]
        df = pd.read_parquet(args.data_dir / info["file"])
        model = sdm.models.KumoTabular(
            task=sdm.Task.regression, size=args.model_size, device=args.device
        )
        top = max(args.queries)
        for context in sorted(
            {min(c, info["rows"] - top) for c in args.contexts}
        ):
            torch._dynamo.reset()
            forwards = Arms(model, args.settings, args.device)
            for query_rows in args.queries:
                for seed in args.seeds:
                    path = (
                        args.out
                        / f"{name}_c{context}_q{query_rows}_s{seed}.json"
                    )
                    if path.exists():
                        continue
                    rows = np.random.default_rng(seed).permutation(len(df))
                    sub = df.iloc[rows[: context + query_rows]]
                    sub = sub.reset_index(drop=True)
                    tables = build_tables(
                        sub, info["target"], True, args.device, context
                    )
                    targets = sub[info["target"]].to_numpy()
                    truth, scale = (
                        targets[context:],
                        float(targets[:context].std()),
                    )
                    preds, launches = {}, {}
                    for arm in arms:
                        preds[arm], launches[arm] = forwards.predict(
                            arm, tables, seed
                        )
                    record = {
                        "dataset": name,
                        "model_size": args.model_size,
                        "context_rows": context,
                        "query_rows": query_rows,
                        "seed": seed,
                        "target_std": scale,
                        "icl_attention_launches": launches,
                        "arms": {
                            arm: {
                                "metrics": metrics(preds[arm], truth),
                                **(
                                    {}
                                    if arm == "compiled"
                                    else {
                                        "vs_compiled": distance(
                                            preds["compiled"],
                                            preds[arm],
                                            scale,
                                        )
                                    }
                                ),
                            }
                            for arm in arms
                        },
                    }
                    path.write_text(json.dumps(record, indent=1))
                    np.savez_compressed(
                        path.with_suffix(".npz"), truth=truth, **preds
                    )
                    print(path.name, flush=True)
                    del tables
                    torch.cuda.empty_cache()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _pair(records, key, setting):
    return " / ".join(
        f"{max(r['arms'][a]['vs_compiled'][key] for r in records):.3f}"
        for a in (setting, "eager")
    )


def _metric(records, key, setting):
    return " / ".join(
        f"{np.mean([r['arms'][a]['metrics'][key] for r in records]):.4g}"
        for a in ("eager", "compiled", setting)
    )


def table(args):
    groups = defaultdict(list)
    for p in sorted(args.results.glob("*.json")):
        r = json.loads(p.read_text())
        groups[(r["dataset"], r["query_rows"])].append(r)
    setting = args.setting
    head = [
        "dataset",
        "query rows",
        "seeds",
        "runs with kernel diff above eager",
        "max quantile diff, kernel / eager",
        "99.9th pct quantile diff, kernel / eager",
        "max mean-prediction diff, kernel / eager",
        "RMSE eager / compiled / kernel",
        "R2 eager / compiled / kernel",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (name, q), rs in sorted(groups.items()):
        over = sum(
            r["arms"][setting]["vs_compiled"]["max_abs_diff"]
            > r["arms"]["eager"]["vs_compiled"]["max_abs_diff"]
            for r in rs
        )
        cells = [
            name,
            q,
            len(rs),
            f"{over} of {len(rs)}",
            _pair(rs, "max_abs_diff", setting),
            _pair(rs, "p999_abs_diff", setting),
            _pair(rs, "point_max", setting),
            _metric(rs, "rmse", setting),
            _metric(rs, "r2", setting),
        ]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data-dir", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--datasets", nargs="+")
    r.add_argument(
        "--settings", nargs="+", default=["sm120"], choices=sorted(EXPECTED)
    )
    r.add_argument(
        "--model-size", default="large", choices=["small", "medium", "large"]
    )
    r.add_argument("--contexts", nargs="+", type=int, default=[50000])
    r.add_argument("--queries", nargs="+", type=int, default=[16, 1024])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--device", default="cuda:0")
    t = sub.add_parser("table")
    t.add_argument("--results", type=Path, required=True)
    t.add_argument("--setting", default="sm120", choices=sorted(EXPECTED))
    t.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        table(args)


if __name__ == "__main__":
    main()
