# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the SM120 kernel modules from ``csrc/`` into ``<out>/<module>.so``.

Headers shared by several modules live in ``csrc/common/`` and are included
as ``"common/<header>"``.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.utils import cpp_extension

CSRC = Path(__file__).parent / "csrc"
CUTLASS_URL = "https://github.com/NVIDIA/cutlass"
CUTLASS_COMMIT = "da5e086dab31d63815acafdac9a9c5893b1c69e2"

_GENCODE = ["-gencode=arch=compute_120,code=sm_120"]
# nvcc's defaults, pinned: the exact kernels' rounding depends on them.
_IEEE = ["--fmad=true", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
_EXACT = (["-O3", *_GENCODE, *_IEEE], ["-O3"])
_ATTENTION = (["-O3", *_GENCODE], ["-O3"])

# module -> (source file, (nvcc flags, host flags), needs CUTLASS)
MODULES = {
    "linear_gelu_sm120": ("linear_gelu.cu", _EXACT, True),
    "gating_sm120": ("gating.cu", _EXACT, True),
    "rope_rms_sm120": ("rope_rms.cu", _EXACT, False),
    "kv_rms_sm120": ("kv_rms.cu", _EXACT, True),
    "query_rms_sm120": ("query_rms.cu", _EXACT, True),
    "projection_rms_sm120": ("projection_rms.cu", _EXACT, True),
    "icl_qk_sm120": ("icl_qk.cu", _EXACT, False),
    "icl_attention_sm120": ("icl_attention.cu", _ATTENTION, False),
    "icl_attention_int8_sm120": ("icl_attention_int8.cu", _EXACT, False),
}


def stamp(module: str) -> str:
    digest = hashlib.sha256()
    paths = sorted((CSRC / module).iterdir())
    if (CSRC / "common").is_dir():
        paths += sorted((CSRC / "common").iterdir())
    for path in paths:
        digest.update(path.name.encode() + path.read_bytes())
    _, flags, _ = MODULES[module]
    digest.update(
        repr(
            (flags, CUTLASS_COMMIT, torch.__version__, torch.version.cuda)
        ).encode()
    )
    return digest.hexdigest()


def check_toolchain() -> None:
    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home is None or not (Path(cuda_home) / "bin" / "nvcc").is_file():
        raise RuntimeError(
            "building kernels needs the CUDA toolkit: set CUDA_HOME to a "
            f"toolkit with bin/nvcc matching torch's CUDA {torch.version.cuda}"
        )
    if shutil.which("ninja") is None:
        raise RuntimeError("building kernels needs ninja (pip install ninja)")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
        12,
        0,
    ):
        raise RuntimeError("building kernels needs a visible SM 12.0 GPU")


def fetch_cutlass(out_dir: Path, cutlass: Path | None) -> Path:
    root = cutlass or (
        Path(os.environ["CUTLASS_ROOT"])
        if "CUTLASS_ROOT" in os.environ
        else None
    )
    root = root or out_dir / "cutlass"
    if not (root / "include" / "cutlass" / "cutlass.h").is_file():
        if cutlass is not None or "CUTLASS_ROOT" in os.environ:
            raise RuntimeError(f"no CUTLASS headers under {root}/include")
        root.mkdir(parents=True, exist_ok=True)
        for command in (
            ["init", "-q"],
            ["fetch", "-q", "--depth", "1", CUTLASS_URL, CUTLASS_COMMIT],
            ["checkout", "-q", "FETCH_HEAD"],
        ):
            subprocess.run(["git", "-C", str(root), *command], check=True)
    return root


def build(
    out_dir: Path,
    modules: Iterable[str] | None = None,
    cutlass: Path | None = None,
    verbose: bool = False,
) -> Path:
    """Build ``modules`` (default all) into ``out_dir``; skip current ones."""
    out_dir = Path(out_dir)
    modules = list(MODULES) if modules is None else list(modules)
    todo = []
    for module in modules:
        marker = out_dir / f"{module}.sha256"
        current = (out_dir / f"{module}.so").is_file() and marker.is_file()
        if not current or marker.read_text() != stamp(module):
            todo.append(module)
    if not todo:
        return out_dir
    check_toolchain()
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    root = (
        fetch_cutlass(out_dir, cutlass)
        if any(MODULES[m][2] for m in todo)
        else None
    )
    for module in todo:
        source, (cuda_flags, host_flags), needs_cutlass = MODULES[module]
        build_dir = out_dir / "build" / module
        build_dir.mkdir(parents=True, exist_ok=True)
        extra = [str(CSRC / module), str(CSRC)] + (
            [str(root / "include")] if needs_cutlass else []
        )
        cpp_extension.load(
            name=module,
            sources=[str(CSRC / module / source)],
            extra_include_paths=extra,
            extra_cflags=host_flags,
            extra_cuda_cflags=cuda_flags,
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=verbose,
        )
        shutil.copyfile(build_dir / f"{module}.so", out_dir / f"{module}.so")
        (out_dir / f"{module}.sha256").write_text(stamp(module))
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.inference.kernels.build"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cutlass",
        type=Path,
        help="CUTLASS checkout; default $CUTLASS_ROOT or a pinned fetch",
    )
    parser.add_argument("--modules", nargs="+", choices=tuple(MODULES))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    build(args.out, args.modules, args.cutlass, verbose=not args.quiet)


if __name__ == "__main__":
    main()
