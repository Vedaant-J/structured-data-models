# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

import sdm
from benchmark.inference import data
from benchmark.inference.config import Config
from benchmark.inference.kernels import integration, registry
from benchmark.inference.schedule import is_optimized, schedule, summarize
from benchmark.inference.snapshot import characterize, compare, leaves
from sdm.models._huggingface import download_checkpoint

BOUNDARY = (
    "model.clear(), gc.collect() and a fresh seeded CUDA generator outside "
    "the clock, Python GC disabled inside it; "
    "clock = CUDA sync, fit(context, labels) + predict(query) under BF16 "
    "autocast and inference mode on CUDA-resident tables, CUDA sync"
)


def set_precision(policy: str) -> None:
    if policy == "ieee":
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            False
        )


def precision_flags() -> dict:
    matmul = torch.backends.cuda.matmul
    return {
        "fp32_precision": torch.backends.fp32_precision,
        "cuda.matmul.fp32_precision": matmul.fp32_precision,
        "cudnn.fp32_precision": torch.backends.cudnn.fp32_precision,
        "cuda.matmul.allow_bf16_reduced_precision_reduction": (
            matmul.allow_bf16_reduced_precision_reduction
        ),
    }


def checkpoint_path(config: Config) -> Path:
    """The checkpoint file to load: ``--checkpoint`` or the pretrained one."""
    if config.checkpoint is not None:
        return config.checkpoint
    return Path(
        download_checkpoint(
            repo_id="nvidia/Kumo-Tabular",
            filename=f"{config.model_size}/classifier.pt",
            revision="v1.0.1",
            config_filename="config.json",
        )
    )


def load_model(config: Config) -> sdm.models.KumoTabular:
    kwargs = {"task": "classification", "size": config.model_size}
    model = sdm.models.KumoTabular(**kwargs, pretrained=False, device="cpu")
    path = checkpoint_path(config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    try:
        model.models[sdm.Task.classification].load_state_dict(state)
    except RuntimeError as error:
        raise SystemExit(
            f"checkpoint {path} is not a {config.model_size} "
            f"KumoTabular classifier:\n{error}"
        ) from error
    return model.to(config.device)


def kernel_counts(prof: profile) -> dict[str, dict]:
    """CUDA kernel name -> launch count and total milliseconds."""
    kernels: dict[str, dict] = {}
    for event in prof.events():
        if event.device_type == DeviceType.CUDA:
            entry = kernels.setdefault(event.name, {"count": 0, "ms": 0.0})
            entry["count"] += 1
            entry["ms"] += event.time_range.elapsed_us() / 1000
    return kernels


def icl_attention_launches(
    census: dict[str, int], kernels: dict[str, dict], expected: str | None
) -> dict[str, dict[str, int | None]]:
    """ICL attention kernel -> registry and profiled CUDA launch counts.

    A kernel made of several CUDA kernels counts its least launched one. A
    kernel sharing the expected kernel's CUDA kernels cannot be told apart
    from it in a profile, so its profiled count is ``None``.
    """
    attention = [k for k in registry.registered() if k.op == "icl_attention"]
    expected_symbols = next(
        (k.cuda_symbols for k in attention if k.name == expected), None
    )
    launches = {}
    for kernel in attention:
        profiled = None
        if kernel.name == expected or kernel.cuda_symbols != expected_symbols:
            profiled = min(
                sum(
                    e["count"] for name, e in kernels.items() if symbol in name
                )
                for symbol in kernel.cuda_symbols
            )
        launches[kernel.name] = {
            "census": census.get(kernel.name, 0),
            "profiled": profiled,
        }
    return launches


def verify_icl_attention(
    launches: dict[str, dict[str, int]], expected: str | None, calls: int
) -> None:
    """Raise unless only ``expected`` launched, ``calls`` times in both."""
    for name, counts in launches.items():
        want = calls if name == expected else 0
        if counts["census"] != want or counts["profiled"] not in (want, None):
            raise RuntimeError(
                f"ICL attention kernel {name} launched {counts}, "
                f"expected {want} (kernel {expected or 'none'})"
            )


class Bench(Protocol):
    unavailable: str | None

    def seconds(self, arm: str) -> float: ...
    def snapshot(self, arm: str) -> dict[str, Tensor]: ...
    def profile(self, arm: str) -> dict[str, dict]: ...


class CudaBench:
    """One model and one set of resident tables shared by all arms."""

    def __init__(self, config: Config) -> None:
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is not available; this benchmark times CUDA work"
            )
        self.device = torch.device(config.device)
        if self.device.type != "cuda":
            raise SystemExit(
                f"--device must be a CUDA device, got {config.device}"
            )
        self.seed = config.seed
        integration.isolate_compiles()
        set_precision(config.precision)
        self.model = load_model(config)
        self.unavailable = None
        self.x, self.y, self.query = data.tables(config)
        self.inner = self.model.models[sdm.Task.classification]
        self.icl_attention_calls = len(self.inner.icl_block.layers) - 1
        self.forwards = {}
        if "compiled" in config.arms:
            self.forwards["compiled"] = torch.compile(
                self.inner.forward, **integration.COMPILE
            )
        self.peak: dict[str, dict] = {}
        for arm in filter(is_optimized, config.arms):
            try:
                self.forwards[arm] = integration.compile_with_kernels(
                    self.inner.forward,
                    config.kernels,
                    config.ops,
                    attention=config.arm_attention(arm),
                    shadow=config.shadow,
                )
            except integration.Unsupported as error:
                self.unavailable = str(error)
                break

    def _run(self, arm: str) -> tuple[float, sdm.TableTensor]:
        self.model.clear()
        generator = torch.Generator(self.device).manual_seed(self.seed)
        with ExitStack() as stack:
            if arm != "eager":
                self.inner.forward = self.forwards[arm]
                stack.callback(vars(self.inner).pop, "forward")
            stack.enter_context(torch.inference_mode())
            stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
            # Collect outside the clock and keep the collector out of it.
            gc.collect()
            gc.disable()
            stack.callback(gc.enable)
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            base = torch.cuda.memory_allocated(self.device)
            start = time.perf_counter()
            self.model.fit(
                x=self.x,
                y=self.y,
                num_estimators=1,
                generator=generator,
            )
            result = self.model.predict(self.query)
            torch.cuda.synchronize(self.device)
            seconds = time.perf_counter() - start
            memory = {
                "base_bytes": base,
                "allocated_bytes": torch.cuda.max_memory_allocated(
                    self.device
                ),
                "reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            }
            peak = self.peak.setdefault(arm, dict.fromkeys(memory, 0))
            for key, value in memory.items():
                peak[key] = max(peak[key], value)
            return seconds, result

    def seconds(self, arm: str) -> float:
        return self._run(arm)[0]

    def snapshot(self, arm: str) -> dict[str, Tensor]:
        _, result = self._run(arm)
        out = leaves(self.model._cache, "cache")
        out["probabilities"] = result.numerical.detach().cpu()
        self.model.clear()
        return out

    def profile(self, arm: str) -> dict[str, dict]:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities) as prof:
            self._run(arm)
        return kernel_counts(prof)


def _no_kernel_admitted(decisions: list[dict]) -> str:
    if not decisions:
        return (
            "no graph region matched any kernel rule; the compiled graph may "
            "differ from the one the rules were written for"
        )
    reasons = sorted(
        {
            f"{name}: {reason}"
            for decision in decisions
            for name, reason in decision["rejected"].items()
        }
    )
    return "no kernel admitted any region on this GPU and configuration" + (
        f" (rejections: {'; '.join(reasons)})" if reasons else ""
    )


def _drop(
    arms: tuple[str, ...], dropped: list[str], reason: str
) -> tuple[str, ...]:
    print(
        f"WARNING: {', '.join(dropped)} skipped, {reason}. "
        "Measuring the other arms only.",
        file=sys.stderr,
    )
    return tuple(arm for arm in arms if arm not in dropped)


def _matching(profile: dict[str, dict], symbols: tuple[str, ...]) -> list[str]:
    return sorted(k for k in profile if any(s in k for s in symbols))


def _kernel_report(
    decisions: list[dict],
    launches: dict[str, int],
    profiles: dict[str, dict],
    arm: str,
) -> dict:
    kernels = {kernel.name: kernel for kernel in registry.registered()}
    verification = {}
    for name in sorted({d["kernel"] for d in decisions} - {"native"}):
        symbols = kernels[name].cuda_symbols
        optimized = _matching(profiles.get(arm, {}), symbols)
        compiled = _matching(profiles.get("compiled", {}), symbols)
        verification[name] = {
            "optimized": optimized,
            "compiled": compiled,
            "verified": not compiled
            and all(any(s in k for k in optimized) for s in symbols),
        }
    return {
        "kernels_verified": bool(verification)
        and all(v["verified"] for v in verification.values()),
        "verification": verification,
        "decisions": decisions,
        "launches_per_optimized_call": launches,
    }


def _gate(decisions: list[dict]) -> str:
    """Gate an optimized arm must pass against compiled before timing.

    ``bitwise`` if every selected kernel is bitwise, ``tolerance`` (distance
    to compiled within compiled vs eager) if some is not, ``opt-in`` (only
    recorded) if some is lossy.
    """
    if not all(d["exact"] for d in decisions):
        return "opt-in"
    if all(d["bitwise"] for d in decisions):
        return "bitwise"
    return "tolerance"


def _passes(entry: dict, scale: dict | None) -> bool:
    if entry["gate"] == "opt-in":
        return True
    if entry["gate"] == "bitwise":
        return entry["optimized_vs_compiled"]["bitwise_equal"]
    distance = entry["optimized_vs_compiled_distance"]
    return (
        distance["probability_max_abs_diff"]
        <= scale["probability_max_abs_diff"]
        and distance["argmax_agreement_pct"] == 100
    )


def _verify_arms(
    report: dict,
    bench: Bench,
    config: Config,
    arms: tuple[str, ...],
    decisions: dict[str, list[dict]],
) -> None:
    calls = getattr(bench, "icl_attention_calls", None)
    if calls is None or not any(
        k.op == "icl_attention" for k in registry.registered()
    ):
        return
    launches = {}
    try:
        for arm in arms:
            chosen = {
                d["kernel"]
                for d in decisions.get(arm, ())
                if d["op"] == "icl_attention"
            } - {"native"}
            expected = None
            if chosen:
                (expected,) = chosen
            expected_calls = calls
            if is_optimized(arm):
                _, options = registry.parse_attention(
                    config.arm_attention(arm)
                )
                if "ends" in options:
                    expected_calls -= 2
            registry.reset_census()
            profiled = bench.profile(arm)
            launches[arm] = icl_attention_launches(
                registry.census()["launches"], profiled, expected
            )
            verify_icl_attention(launches[arm], expected, expected_calls)
    except (RuntimeError, ValueError) as error:
        report["status"] = "launch_verification_failed"
        report["launch_verification_error"] = str(error)
    report["kernels"]["icl_attention_launches"] = launches


def _single(per_arm: dict[str, dict], into: dict) -> dict:
    """With one plain ``optimized`` arm, also keep its entries at the top."""
    if list(per_arm) == ["optimized"]:
        into.update(per_arm["optimized"])
    return into


def measure(config: Config, bench: Bench) -> dict:
    """Warm up, qualify, time and profile every arm of ``bench``."""
    report: dict = {"config": config.record(), "boundary": BOUNDARY}
    arms = config.arms
    if any(map(is_optimized, arms)) and bench.unavailable:
        arms = _drop(
            arms, [a for a in arms if is_optimized(a)], bench.unavailable
        )
        report["optimized_unavailable"] = bench.unavailable
    report["warmup"] = {}
    decisions, shadow = {}, {}
    for arm in arms:
        registry.reset_census()
        times = [bench.seconds(arm) for _ in range(config.warmup)]
        report["warmup"][arm] = {
            "first_call_seconds": times[0],
            "seconds": times,
        }
        if is_optimized(arm):
            decisions[arm] = registry.census()["decisions"]
            shadow[arm] = registry.census()["shadow"]
    native = [
        arm
        for arm, made in decisions.items()
        if all(d["kernel"] == "native" for d in made)
    ]
    if native:
        reason = _no_kernel_admitted(
            [d for arm in native for d in decisions[arm]]
        )
        report["optimized_unavailable"] = reason
        arms = _drop(arms, native, reason)
        for arm in native:
            del decisions[arm], shadow[arm]
    optimized = [arm for arm in arms if is_optimized(arm)]
    report["arms"] = list(arms)
    if config.shadow:
        exact = all(
            e["bitwise"] == e["outputs"]
            for entries in shadow.values()
            for e in entries.values()
        )
        report["status"] = "shadow_bitwise" if exact else "shadow_mismatch"
        per = {
            arm: {"decisions": decisions[arm], "shadow": shadow[arm]}
            for arm in optimized
        }
        report["kernels"] = _single(per, {"arms": per})
        return report

    # Each snapshot is a fresh call; at most two are held at once. Eager is
    # snapshotted for scale whenever a gate is not bitwise, timed or not.
    gates = {arm: _gate(decisions[arm]) for arm in optimized}
    qualification: dict = {
        "gate": _gate([d for made in decisions.values() for d in made]),
        "eager_timed": "eager" in arms,
        "arms": {},
    }
    if "compiled" in arms:
        compiled = bench.snapshot("compiled")
        if "eager" in arms or set(gates.values()) - {"bitwise"}:
            qualification["compiled_vs_eager"] = characterize(
                bench.snapshot("eager"), compiled
            )
        for arm in optimized:
            candidate = bench.snapshot(arm)
            entry = {
                "gate": gates[arm],
                "optimized_vs_compiled": compare(compiled, candidate),
                "optimized_vs_compiled_distance": characterize(
                    compiled, candidate
                ),
            }
            del candidate
            entry["passed"] = _passes(
                entry, qualification.get("compiled_vs_eager")
            )
            qualification["arms"][arm] = entry
        del compiled
    passed = all(e["passed"] for e in qualification["arms"].values())
    _single(qualification["arms"], qualification)
    qualification["passed"] = passed
    report["qualification"] = qualification
    if config.qualify_only or (not passed and not config.allow_mismatch):
        report["status"] = "qualified" if passed else "qualification_failed"
        per = {arm: {"decisions": decisions[arm]} for arm in optimized}
        report["kernels"] = _single(per, {"arms": per})
        if config.qualify_only:
            _verify_arms(report, bench, config, arms, decisions)
        return report

    # Warmup compiled every graph; a recompile inside the clock is an error.
    peak = getattr(bench, "peak", {})
    peak.clear()
    start = time.time()
    with torch._dynamo.config.patch(error_on_recompile=True):
        samples = [
            {
                "block": block,
                "position": position,
                "arm": arm,
                "seconds": bench.seconds(arm),
            }
            for block, order in enumerate(schedule(arms, config.blocks))
            for position, arm in enumerate(order)
        ]
    report["timing_window_unix"] = [start, time.time()]
    report["peak_memory"] = {arm: dict(v) for arm, v in peak.items()}
    report["samples"] = samples
    report["statistics"] = summarize(samples, arms)

    profiles, launches = {}, {}
    for arm in arms:
        registry.reset_census()
        profiles[arm] = bench.profile(arm)
        launches[arm] = registry.census()["launches"]
    per = {
        arm: _kernel_report(decisions[arm], launches[arm], profiles, arm)
        for arm in optimized
    }
    report["kernels"] = _single(per, {"arms": per, "profiles": profiles})
    if not passed:
        report["status"] = "completed_with_mismatch"
    elif "optimized_unavailable" in report:
        report["status"] = "completed_without_optimized"
    else:
        report["status"] = "completed"
    return report
