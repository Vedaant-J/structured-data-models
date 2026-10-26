# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

from benchmark.inference.config import parse
from benchmark.inference.kernels import integration
from benchmark.inference.report import environment, summary, write
from benchmark.inference.runner import (
    CudaBench,
    checkpoint_path,
    measure,
    precision_flags,
)

EXIT_CODES = {
    "completed": 0,
    "qualified": 0,
    "shadow_bitwise": 0,
    "completed_without_optimized": 2,
}


def main(argv: list[str] | None = None) -> int:
    config = parse(argv)
    bench = CudaBench(config)
    report = measure(config, bench)
    with checkpoint_path(config).open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    report["environment"] = {
        **environment(bench.device),
        "checkpoint_sha256": digest,
        "precision": precision_flags(),
        "compile": integration.COMPILE,
    }
    write(config.output, report)
    print(summary(report))
    print(f"Wrote {config.output}")
    return EXIT_CODES.get(report["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
