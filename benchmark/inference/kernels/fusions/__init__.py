# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One module per kernel family.

Importing a module defines its ops and graph rules; ``register(directory)``
registers its kernels from the built modules in ``directory``.
"""

import importlib
import pkgutil
from pathlib import Path


def load(directory: Path) -> None:
    names = sorted(info.name for info in pkgutil.iter_modules(__path__))
    # Import every module first: gating.py registers a linear_gelu kernel.
    modules = [importlib.import_module(f"{__name__}.{n}") for n in names]
    for module in modules:
        module.register(directory)
