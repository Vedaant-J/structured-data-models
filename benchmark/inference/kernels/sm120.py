# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the SM120 (Blackwell) kernel registrations."""

import importlib.machinery
import importlib.util
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor

# Below this many rows cuBLAS can pick another algorithm, whose summation
# order the GEMM kernels do not reproduce, so small (query-side) calls stay
# native.
MIN_M = 1024
EPS = 2.0**-23  # the RMS kernels hard-code finfo(float32).eps


def load_module(directory: Path, name: str) -> ModuleType:
    path = directory / f"{name}.so"
    if not path.is_file():
        raise FileNotFoundError(f"missing built kernel module {path}")
    loader = importlib.machinery.ExtensionFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def policy(*tensors: Tensor) -> str | None:
    if any(t.device.type != "cuda" for t in tensors):
        return "needs CUDA tensors"
    if torch.cuda.get_device_capability(tensors[0].device) != (12, 0):
        return "needs SM 12.0"
    if any(t.dtype != torch.bfloat16 for t in tensors):
        return "needs bfloat16"
    if any(not t.is_contiguous() or t.storage_offset() for t in tensors):
        return "needs contiguous, offset-zero operands"
    if torch.backends.cuda.matmul.fp32_precision == "tf32":
        return "needs TF32 disabled"
    if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        return "needs reduced-precision BF16 reductions disabled"
    return None
