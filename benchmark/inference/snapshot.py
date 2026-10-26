# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections.abc import Mapping

import torch
from torch import Tensor

_INT_OF_SIZE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def leaves(value: object, path: str = "") -> dict[str, Tensor]:
    """CPU copies of tensors nested in mappings, sequences, dataclasses."""
    if isinstance(value, Tensor) and hasattr(value, "__tensor_flatten__"):
        # Wrapper subclasses (e.g. SDM's StringTensor): compare their parts.
        names, _ = value.__tensor_flatten__()
        items = ((name, getattr(value, name)) for name in names)
        return {
            k: v
            for name, item in items
            for k, v in leaves(item, f"{path}/{name}").items()
        }
    if isinstance(value, Tensor):
        return {path: value.detach().cpu()}
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, list | tuple):
        items = enumerate(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        items = (
            (f.name, getattr(value, f.name)) for f in dataclasses.fields(value)
        )
    else:
        return {}
    out: dict[str, Tensor] = {}
    for key, item in items:
        out.update(leaves(item, f"{path}/{key}"))
    return out


def _max_abs_diff(a: Tensor, b: Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return (a.double() - b.double()).abs().max().item()


def _differing(a: Tensor, b: Tensor) -> int:
    bits = _INT_OF_SIZE[a.element_size()]
    return int((a.reshape(-1).view(bits) != b.reshape(-1).view(bits)).sum())


def compare(
    reference: dict[str, Tensor], candidate: dict[str, Tensor]
) -> dict:
    """Bitwise comparison of two snapshots, leaf by leaf."""
    mismatches = []
    for path in [*reference, *(candidate.keys() - reference.keys())]:
        a, b = reference.get(path), candidate.get(path)
        if a is None or b is None:
            side = "reference" if a is None else "candidate"
            mismatches.append({"path": path, "reason": f"missing in {side}"})
        elif a.dtype != b.dtype or a.shape != b.shape:
            mismatches.append(
                {
                    "path": path,
                    "reason": f"{a.dtype}{list(a.shape)} vs "
                    f"{b.dtype}{list(b.shape)}",
                }
            )
        elif count := _differing(a, b):
            mismatches.append(
                {
                    "path": path,
                    "differing_elements": count,
                    "elements": a.numel(),
                    "max_abs_diff": _max_abs_diff(a, b),
                }
            )
    return {
        "bitwise_equal": not mismatches,
        "leaves": len(reference),
        "mismatched_leaves": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
    }


def characterize(
    reference: dict[str, Tensor],
    candidate: dict[str, Tensor],
) -> dict:
    """Numerical distance of ``candidate`` from ``reference``."""
    p = reference["probabilities"].double()
    q = candidate["probabilities"].double()
    diff = (p - q).abs()
    return {
        "probability_max_abs_diff": diff.max().item(),
        "probability_mean_abs_diff": diff.mean().item(),
        "argmax_agreement_pct": 100
        * (p.argmax(-1) == q.argmax(-1)).double().mean().item(),
        "leaf_max_abs_diff": {
            path: _max_abs_diff(a, candidate[path])
            for path, a in reference.items()
            if path in candidate and a.shape == candidate[path].shape
        },
    }
