# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task metrics of ICL attention settings against compiled SDM.

    python -m benchmark.attention.compare run --data-dir D --out R \
        --settings sm120 int8 int8+ends --contexts 50000 --queries 16 1024 \
        --seeds 0 1 2
    python -m benchmark.attention.compare tabulate --results R

Each setting arm is compiled SDM plus only the ``icl_attention`` op under
that ``SDM_ICL_ATTENTION`` value. One JSON (and one npz of predictions) per
dataset, model size, context, query rows and seed, with the eager, compiled
and every setting arm. ``--data-dir`` is where
``benchmark.attention.compare_data`` wrote its tables.
"""

import argparse
import gc
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, r2_score, roc_auc_score

import sdm
from benchmark.inference import runner
from benchmark.inference.kernels import integration, registry

KERNEL_OF_BACKEND = {
    "sm120": "sm120_icl_attention_d64",
    "fa2": "fa2_icl_attention_d64",
    "int8": "sm120_int8_icl",
}
CLIP = 1e-7
# More classes than this run ECOC codes as an ICL batch above 1, which the
# 8-bit kernel does not take (batch 1 only).
MAX_INT8_CLASSES = 10
# LEVELS[i] is the (i + 1) / 1000 quantile.
LEVELS = (np.arange(999) + 1) / 1000
MEDIAN = 499
BANDS = {
    "low_tail": LEVELS <= 0.05,
    "median": (LEVELS >= 0.45) & (LEVELS <= 0.55),
    "high_tail": LEVELS >= 0.95,
}


def build_tables(
    df: pd.DataFrame, target: str, regression: bool, device: str, context: int
) -> tuple[sdm.TableTensor, sdm.TableTensor, sdm.TableTensor]:
    overrides = {} if regression else {target: "categorical"}
    stypes = sdm.infer_stypes(df, overrides=overrides)
    table = sdm.TableTensor.from_pandas(df, stypes=stypes)
    features, labels = table.drop_columns(target), table[:, target]
    return (
        features[:context].to(device),
        labels[:context].to(device),
        features[context:].to(device),
    )


class Arms:
    def __init__(self, model, task, settings, device):
        self.model, self.device = model, device
        self.expected = {
            s: KERNEL_OF_BACKEND.get(registry.parse_attention(s)[0])
            for s in settings
        }
        self.inner = model.models[task]
        self.forwards = {
            "compiled": torch.compile(
                self.inner.forward, **integration.COMPILE
            )
        }
        for setting in settings:
            self.forwards[setting] = integration.compile_with_kernels(
                self.inner.forward, ops=("icl_attention",), attention=setting
            )

    def predict(self, arm, tables, seed):
        x, y, query = tables
        self.model.clear()
        generator = torch.Generator(self.device).manual_seed(seed)
        with ExitStack() as stack:
            if arm != "eager":
                self.inner.forward = self.forwards[arm]
                stack.callback(vars(self.inner).pop, "forward")
            stack.enter_context(torch.inference_mode())
            stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
            gc.collect()
            torch.cuda.synchronize(self.device)
            before = dict(registry.census()["launches"])
            start = time.perf_counter()
            self.model.fit(x=x, y=y, num_estimators=1, generator=generator)
            result = self.model.predict(query)
            torch.cuda.synchronize(self.device)
            seconds = time.perf_counter() - start
        after = registry.census()["launches"]
        launches = {
            k: v - before.get(k, 0)
            for k, v in after.items()
            if v != before.get(k, 0)
        }
        expected = self.expected.get(arm)
        if set(launches) != ({expected} if expected else set()):
            raise RuntimeError(
                f"arm {arm!r} launched {launches}, expected {expected}"
            )
        out = result.numerical.detach().float().cpu().numpy()
        names = result.columns[sdm.Stype.numerical]
        self.model.clear()
        return out, names, seconds, launches


def classification_metrics(prob, classes, truth, num_classes):
    full = np.zeros((len(prob), num_classes))
    full[:, classes] = prob
    full /= full.sum(1, keepdims=True)
    full = np.clip(full, CLIP, 1)
    full /= full.sum(1, keepdims=True)
    labels = np.arange(num_classes)
    metrics = {
        "accuracy": float((full.argmax(1) == truth).mean()),
        "log_loss": float(log_loss(truth, full, labels=labels)),
        "auc": None,
    }
    present = np.unique(truth)
    if num_classes == 2 and len(present) == 2:
        metrics["auc"] = float(roc_auc_score(truth, full[:, 1]))
    elif num_classes > 2 and len(present) == num_classes:
        metrics["auc"] = float(
            roc_auc_score(
                truth, full, multi_class="ovr", average="macro", labels=labels
            )
        )
    return metrics


def regression_metrics(quantiles, truth, scale):
    quantiles = np.sort(quantiles.astype(np.float64), axis=1)
    mean, median = quantiles.mean(1), quantiles[:, MEDIAN]
    err = truth[:, None] - quantiles
    pinball = np.maximum(LEVELS * err, (LEVELS - 1) * err)
    metrics = {
        "rmse": float(np.sqrt(((mean - truth) ** 2).mean())),
        "r2": float(r2_score(truth, mean)),
        "rmse_median": float(np.sqrt(((median - truth) ** 2).mean())),
        "pinball": float(pinball.mean()),
        "pinball_q05": float(pinball[:, 49].mean()),
        "pinball_q50": float(pinball[:, MEDIAN].mean()),
        "pinball_q95": float(pinball[:, 949].mean()),
    }
    for name, lo, hi in (("80", 99, 899), ("90", 49, 949)):
        inside = (truth >= quantiles[:, lo]) & (truth <= quantiles[:, hi])
        metrics[f"coverage_{name}"] = float(inside.mean())
        metrics[f"width_{name}"] = float(
            (quantiles[:, hi] - quantiles[:, lo]).mean() / scale
        )
    return metrics


def distance(reference, candidate, regression, scale):
    ref, cand = reference.astype(np.float64), candidate.astype(np.float64)
    diff = np.abs(ref - cand)
    if regression:
        diff = diff / scale
        out = {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
        }
        for band, mask in BANDS.items():
            out[f"{band}_max"] = float(diff[:, mask].max())
            out[f"{band}_mean"] = float(diff[:, mask].mean())
        point = np.abs(ref.mean(1) - cand.mean(1)) / scale
        out.update(
            point_max=float(point.max()), point_mean=float(point.mean())
        )
        return out
    flips = int((ref.argmax(1) != cand.argmax(1)).sum())
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "flips": flips,
        "flip_rate": flips / len(ref),
    }


def run_group(args, name, info, df, size, context, model, task):
    regression = info["task"] == "regression"
    settings = [
        s
        for s in args.settings
        if (info["classes"] or 0) <= MAX_INT8_CLASSES
        or registry.parse_attention(s)[0] != "int8"
    ]
    arms = ["eager", "compiled", *settings]
    torch._dynamo.reset()
    forwards = Arms(model, task, settings, args.device)
    for query_rows in args.queries:
        for seed in args.seeds:
            path = (
                args.out
                / f"{name}_{size}_c{context}_q{query_rows}_s{seed}.json"
            )
            if path.exists():
                continue
            rows = np.random.default_rng(seed).permutation(len(df))[
                : context + query_rows
            ]
            sub = df.iloc[rows].reset_index(drop=True)
            tables = build_tables(
                sub, info["target"], regression, args.device, context
            )
            targets = sub[info["target"]].to_numpy()
            truth = targets[context:]
            scale = float(targets[:context].std())
            preds, seconds, columns, launches = {}, {}, {}, {}
            for arm in arms:
                preds[arm], columns[arm], seconds[arm], launches[arm] = (
                    forwards.predict(arm, tables, seed)
                )
            record = {
                "dataset": name,
                "task": info["task"],
                "model_size": size,
                "context_rows": context,
                "query_rows": query_rows,
                "seed": seed,
                "features": info["features"],
                "classes": info["classes"],
                "target_std": scale,
                "seconds": seconds,
                "launches": launches,
                "arms": {},
            }
            for arm in arms:
                if regression:
                    metrics = regression_metrics(preds[arm], truth, scale)
                else:
                    classes = np.array([int(c) for c in columns[arm]])
                    metrics = classification_metrics(
                        preds[arm], classes, truth, info["classes"]
                    )
                entry = {"metrics": metrics}
                if arm != "compiled":
                    entry["vs_compiled"] = distance(
                        preds["compiled"], preds[arm], regression, scale
                    )
                record["arms"][arm] = entry
            path.write_text(json.dumps(record, indent=1))
            np.savez_compressed(path.with_suffix(".npz"), truth=truth, **preds)
            gap = record["arms"]["eager"]["vs_compiled"]["max_abs_diff"]
            times = {k: round(v, 1) for k, v in seconds.items()}
            print(
                f"{path.name} eager_gap={gap:.3e} seconds={times}", flush=True
            )
            del tables
            torch.cuda.empty_cache()


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    runner.set_precision("ieee")
    for name in args.datasets or manifest:
        info = manifest[name]
        df = pd.read_parquet(args.data_dir / info["file"])
        task = (
            sdm.Task.regression
            if info["task"] == "regression"
            else sdm.Task.classification
        )
        for size in args.model_sizes:
            model = sdm.models.KumoTabular(
                task=task, size=size, device=args.device
            )
            top = max(args.queries)
            contexts = sorted(
                {min(c, info["rows"] - top) for c in args.contexts}
            )
            for context in contexts:
                run_group(args, name, info, df, size, context, model, task)
            del model
            gc.collect()
            torch.cuda.empty_cache()


def _cell(values, fmt):
    return f"{fmt % np.mean(values)} / {fmt % max(values)}"


def _metric(records, arm, key):
    values = [r["arms"][arm]["metrics"][key] for r in records]
    if None in values:
        return "-"
    base = [r["arms"]["compiled"]["metrics"][key] for r in records]
    return f"{np.mean(values):.4f} ({np.mean(values) - np.mean(base):+.4f})"


def _stat(records, arm, key):
    return [r["arms"][arm]["vs_compiled"][key] for r in records]


def _table(title, head, rows):
    out = [
        f"### {title}\n",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return [*out, ""]


def tabulate(args):
    groups = defaultdict(list)
    for path in sorted(args.results.glob("*.json")):
        record = json.loads(path.read_text())
        key = (
            record["task"],
            record["model_size"],
            record["context_rows"],
            record["query_rows"],
            record["dataset"],
        )
        groups[key].append(record)
    sections = defaultdict(lambda: ([], []))
    for (task, size, context, query_rows, name), records in sorted(
        groups.items()
    ):
        for arm in records[0]["arms"]:
            base = [name, arm, str(len(records))]
            moved = arm != "compiled"
            if task == "regression":
                bands = ["low_tail", "median", "high_tail"]
                metric = [
                    _metric(records, arm, k)
                    for k in (
                        "rmse",
                        "r2",
                        "pinball",
                        "coverage_80",
                        "width_80",
                        "coverage_90",
                        "width_90",
                    )
                ]
                move = (
                    [
                        _cell(_stat(records, arm, f"{b}_{m}"), "%.4f")
                        for b in bands
                        for m in ("max", "mean")
                    ]
                    + [_cell(_stat(records, arm, "point_max"), "%.4f")]
                    if moved
                    else ["-"] * 7
                )
            else:
                metric = [
                    _metric(records, arm, k)
                    for k in ("accuracy", "auc", "log_loss")
                ]
                move = (
                    [
                        _cell(_stat(records, arm, "max_abs_diff"), "%.4f"),
                        _cell(_stat(records, arm, "mean_abs_diff"), "%.5f"),
                        _cell(_stat(records, arm, "flips"), "%.1f"),
                        "%.2f%%"
                        % (100 * np.mean(_stat(records, arm, "flip_rate"))),
                    ]
                    if moved
                    else ["-"] * 4
                )
            sections[(task, size, context, query_rows)][0].append(
                base + metric
            )
            sections[(task, size, context, query_rows)][1].append(base + move)
    out = []
    for (task, size, context, query_rows), (
        metric_rows,
        move_rows,
    ) in sections.items():
        label = f"{task}, {size}, context {context}, Q{query_rows}"
        head = ["dataset", "arm", "seeds"]
        if task == "regression":
            out += _table(
                f"{label}: task metrics, mean over seeds (change vs compiled)",
                [
                    *head,
                    "RMSE",
                    "R2",
                    "pinball",
                    "coverage 80",
                    "width 80/std",
                    "coverage 90",
                    "width 90/std",
                ],
                metric_rows,
            )
            out += _table(
                f"{label}: quantile movement vs compiled, |d|/target std, "
                "mean / worst over seeds",
                [
                    *head,
                    "low tail max",
                    "low tail mean",
                    "median max",
                    "median mean",
                    "high tail max",
                    "high tail mean",
                    "mean-pred max",
                ],
                move_rows,
            )
        else:
            out += _table(
                f"{label}: task metrics, mean over seeds (change vs compiled)",
                [*head, "accuracy", "AUC", "log-loss"],
                metric_rows,
            )
            out += _table(
                f"{label}: probability movement vs compiled, "
                "mean / worst over seeds",
                [*head, "max |dp|", "mean |dp|", "flips", "flip rate"],
                move_rows,
            )
    text = "\n".join(out)
    print(text)
    if args.output:
        args.output.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data-dir", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--datasets", nargs="+")
    r.add_argument("--settings", nargs="*", default=[])
    r.add_argument(
        "--model-sizes",
        nargs="+",
        default=["large"],
        choices=["small", "medium", "large"],
    )
    r.add_argument("--contexts", nargs="+", type=int, default=[50000])
    r.add_argument("--queries", nargs="+", type=int, default=[16, 1024])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--device", default="cuda:0")
    t = sub.add_parser("tabulate")
    t.add_argument("--results", type=Path, required=True)
    t.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        tabulate(args)


if __name__ == "__main__":
    main()
