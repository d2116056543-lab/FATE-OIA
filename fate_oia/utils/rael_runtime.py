"""Strict runtime profiling for real RAEL public trainer adapters.

The profiler deliberately owns measurement and artifact persistence only.  A
caller must supply a factory for a real trainer adapter; this module never
constructs data, models, optimizers, or synthetic updates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from fate_oia.utils.rael_artifacts import _validate_run_json, _validate_run_jsonl_row


WARMUP_OPTIMIZER_UPDATES = 5
MEASURED_OPTIMIZER_UPDATES = 20
MAX_RESERVED_GB = 45.0
ELIGIBLE_RESERVED_GB = 42.0
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class RuntimeCandidate:
    name: str
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def payload(self) -> dict[str, int | str]:
        return asdict(self)


CANDIDATES = (
    RuntimeCandidate("P0", 8, 4, 8),
    RuntimeCandidate("P1", 6, 5, 8),
    RuntimeCandidate("P2", 4, 8, 8),
)

CORE_MECHANISM_FLAGS = {
    "dino": True,
    "four_layers": True,
    "slot": True,
    "unary": True,
    "pairwise": True,
    "gradient_admission": True,
    "mirror_25pct": True,
    "counterfactual": False,
}
RUNTIME_STEP_MECHANISM_FLAGS = {
    **CORE_MECHANISM_FLAGS,
    # P18's runtime row schema names the slot/unary/admission composite ledger.
    "ledger": True,
}


class NoEligibleRuntimeProfile(RuntimeError):
    """Raised after artifacts are written when no candidate meets the hard gate."""


@runtime_checkable
class RuntimeRunner(Protocol):
    """Public trainer adapter protocol consumed by the runtime profiler.

    ``optimizer_update`` must execute one real optimizer update, including the
    configured gradient accumulation, and return measurements observed by the
    RAEL trainer.  ``measure_counterfactual_overhead`` must execute its real
    public counterfactual measurement path independently of the core update.
    """

    def artifact_provenance(self) -> Mapping[str, Any]: ...

    def configure_runtime_profile(self, context: Mapping[str, Any]) -> None: ...

    def optimizer_update(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def measure_counterfactual_overhead(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


class MemoryProbe(Protocol):
    def before_update(self, device: str) -> None: ...

    def after_update(self, device: str) -> tuple[float, float]: ...


class _CudaMemoryProbe:
    """Production memory probe backed by torch.cuda peak statistics."""

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - depends on deployment.
            raise RuntimeError("torch is required for CUDA runtime profiling") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime profiling requires an available torch.cuda device")
        self._torch = torch

    def before_update(self, device: str) -> None:
        self._torch.cuda.synchronize(device)
        self._torch.cuda.reset_peak_memory_stats(device)

    def after_update(self, device: str) -> tuple[float, float]:
        self._torch.cuda.synchronize(device)
        gib = float(1024**3)
        return (
            float(self._torch.cuda.max_memory_allocated(device)) / gib,
            float(self._torch.cuda.max_memory_reserved(device)) / gib,
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _finite_number(value: Any, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a numeric scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    if nonnegative and resolved < 0:
        raise ValueError(f"{field} must be nonnegative")
    return resolved


def _validate_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    required = ("schema_version", "producer", "source_fingerprint_sha256", "config_sha256")
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"P18 provenance is missing fields: {missing}")
    if value["schema_version"] != "rael-artifact-v1":
        raise ValueError("P18 provenance schema_version must be rael-artifact-v1")
    if not isinstance(value["producer"], str) or not value["producer"].strip():
        raise ValueError("P18 provenance producer must be nonempty")
    for field in ("source_fingerprint_sha256", "config_sha256"):
        if not isinstance(value[field], str) or not _SHA256.fullmatch(value[field]):
            raise ValueError(f"P18 provenance {field} must be a lowercase SHA-256")
    return {field: value[field] for field in required}


def _runner_provenance(runner: Any) -> dict[str, Any]:
    method = getattr(runner, "artifact_provenance", None)
    if not callable(method):
        raise TypeError("runner must expose artifact_provenance() for P18 artifact binding")
    provenance = method()
    if not isinstance(provenance, Mapping):
        raise TypeError("runner artifact_provenance() must return a mapping")
    return _validate_provenance(provenance)


def _require_runner_protocol(runner: Any) -> None:
    for method in ("configure_runtime_profile", "optimizer_update", "measure_counterfactual_overhead"):
        if not callable(getattr(runner, method, None)):
            raise TypeError(f"runner must expose real public trainer method {method}()")


def _core_context(candidate: RuntimeCandidate, *, phase: str, optimizer_update: int) -> dict[str, Any]:
    return {
        "candidate": candidate.name,
        "batch_size": candidate.batch_size,
        "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        "num_workers": candidate.num_workers,
        "effective_batch_size": candidate.effective_batch_size,
        "phase": phase,
        "optimizer_update": optimizer_update,
        "mechanism_flags": dict(CORE_MECHANISM_FLAGS),
        "counterfactual": False,
    }


def _counterfactual_context(candidate: RuntimeCandidate) -> dict[str, Any]:
    context = _core_context(candidate, phase="counterfactual_overhead", optimizer_update=0)
    context["counterfactual"] = True
    flags = dict(CORE_MECHANISM_FLAGS)
    flags["counterfactual"] = True
    context["mechanism_flags"] = flags
    return context


def _normalize_mechanism_flags(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise ValueError("mechanism_flags must be a mapping")
    expected = set(CORE_MECHANISM_FLAGS)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        raise ValueError(f"mechanism_flags must contain exactly {sorted(expected)}; missing={missing}, unexpected={unexpected}")
    flags: dict[str, bool] = {}
    for name in CORE_MECHANISM_FLAGS:
        if not isinstance(value[name], bool):
            raise ValueError(f"mechanism_flags.{name} must be bool")
        flags[name] = value[name]
    disabled = [name for name, expected in CORE_MECHANISM_FLAGS.items() if flags[name] != expected]
    if disabled:
        raise ValueError(f"mechanism flags violate core profile: {disabled}")
    # P18 calls the slot/unary/gradient-admission composite its ledger.
    flags["ledger"] = flags["slot"] and flags["unary"] and flags["gradient_admission"]
    if flags != RUNTIME_STEP_MECHANISM_FLAGS:
        raise ValueError("mechanism_flags violate the exact runtime-step schema")
    return flags


def _measure(
    operation: Callable[[], Mapping[str, Any]],
    *,
    device: str,
    clock: Callable[[], float],
    memory_probe: MemoryProbe,
) -> tuple[Mapping[str, Any], float, float, float]:
    memory_probe.before_update(device)
    started = _finite_number(clock(), field="clock start")
    result = operation()
    allocated_gb, reserved_gb = memory_probe.after_update(device)
    # CUDA probes synchronize in after_update, so this end timestamp includes device work.
    finished = _finite_number(clock(), field="clock end")
    elapsed = finished - started
    if elapsed <= 0:
        raise ValueError("elapsed must be positive")
    return (
        result,
        elapsed,
        _finite_number(allocated_gb, field="allocated_gb", nonnegative=True),
        _finite_number(reserved_gb, field="reserved_gb", nonnegative=True),
    )


def _measured_row(
    metrics: Mapping[str, Any],
    *,
    candidate: RuntimeCandidate,
    update: int,
    elapsed: float,
    allocated_gb: float,
    reserved_gb: float,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    samples = metrics.get("samples")
    expected_samples = candidate.effective_batch_size
    if samples != expected_samples or isinstance(samples, bool):
        reasons.append(f"samples must equal batch_size*grad_accum ({expected_samples})")
        samples = 0
    microbatches = metrics.get("microbatches")
    if isinstance(microbatches, bool) or microbatches != candidate.gradient_accumulation_steps:
        reasons.append("gradient_accumulation_steps mismatch")
        microbatches = 0
    finite = metrics.get("finite")
    if finite is not True:
        reasons.append("finite must be true")
    dino_call_count = metrics.get("dino_call_count_per_microbatch")
    if dino_call_count != 1:
        reasons.append("dino_call_count_per_microbatch must equal 1")
    if isinstance(dino_call_count, bool) or not isinstance(dino_call_count, int):
        dino_call_count = 0
    dino_call_count_total = metrics.get("dino_call_count_total")
    if isinstance(dino_call_count_total, bool) or dino_call_count_total != microbatches:
        reasons.append("dino_call_count_total must equal microbatches")
    if isinstance(dino_call_count_total, bool) or not isinstance(dino_call_count_total, int):
        dino_call_count_total = 0
    try:
        flags = _normalize_mechanism_flags(metrics.get("mechanism_flags"))
    except ValueError as error:
        reasons.append(f"mechanism: {error}")
        flags = {**CORE_MECHANISM_FLAGS, "ledger": False}
    try:
        data_time = _finite_number(metrics.get("data_time"), field="data_time", nonnegative=True)
        dino_time = _finite_number(metrics.get("dino_time"), field="dino_time", nonnegative=True)
    except ValueError as error:
        reasons.append(str(error))
        data_time = 0.0
        dino_time = 0.0
    if reserved_gb >= MAX_RESERVED_GB:
        reasons.append("reserved>=45GB")
    if reserved_gb > ELIGIBLE_RESERVED_GB:
        reasons.append("reserved>42GB")
    samples_per_sec = float(samples) / elapsed if samples else 0.0
    if samples_per_sec <= 0 or not math.isfinite(samples_per_sec):
        reasons.append("samples_per_sec must be positive")
    row = {
        **provenance,
        "epoch": 0,
        "candidate": candidate.name,
        "microbatch_step": update * candidate.gradient_accumulation_steps,
        "optimizer_step": update,
        "optimizer_update": update,
        "elapsed": elapsed,
        "data_time": data_time,
        "dino_time": dino_time,
        "step_time": elapsed,
        "samples": samples,
        "microbatches": microbatches,
        "samples_per_sec": samples_per_sec,
        "allocated_gb": allocated_gb,
        "reserved_gb": reserved_gb,
        "dino_call_count": dino_call_count,
        "dino_call_count_total": dino_call_count_total,
        "finite": finite is True,
        "mechanism_flags": flags,
    }
    return row, reasons


def _counterfactual_overhead(
    runner: RuntimeRunner,
    candidate: RuntimeCandidate,
    *,
    device: str,
    clock: Callable[[], float],
    memory_probe: MemoryProbe,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    payload: dict[str, Any] = {
        **provenance,
        "candidate": candidate.name,
        "elapsed": None,
        "samples": None,
        "samples_per_sec": None,
        "counterfactual_executed": None,
        "available": None,
        "valid_target_count": None,
        "dino_call_count": None,
        "finite": None,
        "loss": None,
    }
    try:
        metrics, elapsed, _, _ = _measure(
            lambda: runner.measure_counterfactual_overhead(_counterfactual_context(candidate)),
            device=device,
            clock=clock,
            memory_probe=memory_probe,
        )
        samples = metrics.get("samples")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise ValueError("counterfactual samples must be a positive integer")
        payload["counterfactual_executed"] = metrics.get("counterfactual_executed")
        if payload["counterfactual_executed"] is not True:
            raise ValueError("counterfactual_executed must be true")
        payload["available"] = metrics.get("available")
        if payload["available"] is not True:
            raise ValueError("available must be true")
        payload["valid_target_count"] = metrics.get("valid_target_count")
        if (
            isinstance(payload["valid_target_count"], bool)
            or not isinstance(payload["valid_target_count"], int)
            or payload["valid_target_count"] <= 0
        ):
            raise ValueError("valid_target_count must be a positive integer")
        payload["dino_call_count"] = metrics.get("dino_call_count")
        if payload["dino_call_count"] != 0:
            raise ValueError("dino_call_count must equal 0")
        payload["finite"] = metrics.get("finite")
        if metrics.get("finite") is not True:
            raise ValueError("counterfactual finite must be true")
        payload["loss"] = _finite_number(metrics.get("loss"), field="counterfactual loss")
        payload.update({"elapsed": elapsed, "samples": samples, "samples_per_sec": samples / elapsed})
        return payload, None
    except Exception as error:
        return payload, f"counterfactual_overhead: {error}"


def _profile_candidate(
    runner_factory: Callable[..., RuntimeRunner],
    candidate: RuntimeCandidate,
    *,
    device: str,
    clock: Callable[[], float],
    memory_probe: MemoryProbe,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    runner = runner_factory(candidate=candidate.payload(), device=device)
    _require_runner_protocol(runner)
    provenance = _runner_provenance(runner)
    runner.configure_runtime_profile(_core_context(candidate, phase="configure", optimizer_update=0))
    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    for update in range(WARMUP_OPTIMIZER_UPDATES):
        try:
            runner.optimizer_update(_core_context(candidate, phase="warmup", optimizer_update=update))
        except Exception as error:
            reasons.append(f"oom_or_update_failure: {error}")
            break
    if not reasons:
        for update in range(MEASURED_OPTIMIZER_UPDATES):
            try:
                metrics, elapsed, allocated_gb, reserved_gb = _measure(
                    lambda update=update: runner.optimizer_update(_core_context(candidate, phase="measured", optimizer_update=update)),
                    device=device,
                    clock=clock,
                    memory_probe=memory_probe,
                )
                row, update_reasons = _measured_row(
                    metrics,
                    candidate=candidate,
                    update=update,
                    elapsed=elapsed,
                    allocated_gb=allocated_gb,
                    reserved_gb=reserved_gb,
                    provenance=provenance,
                )
                rows.append(row)
                reasons.extend(update_reasons)
            except Exception as error:
                reasons.append(f"oom_or_update_failure: {error}")
                break
    overhead, overhead_reason = _counterfactual_overhead(
        runner, candidate, device=device, clock=clock, memory_probe=memory_probe, provenance=provenance
    )
    if overhead_reason:
        reasons.append(overhead_reason)
    if len(rows) != MEASURED_OPTIMIZER_UPDATES:
        reasons.append("did not complete 20 measured optimizer updates")
    total_elapsed = sum(row["elapsed"] for row in rows)
    total_samples = sum(row["samples"] for row in rows if isinstance(row["samples"], int))
    candidate_payload: dict[str, Any] = {
        "name": candidate.name,
        "batch_size": candidate.batch_size,
        "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        "num_workers": candidate.num_workers,
        "effective_batch_size": candidate.effective_batch_size,
        "warmup_optimizer_updates": WARMUP_OPTIMIZER_UPDATES,
        "measured_optimizer_updates": MEASURED_OPTIMIZER_UPDATES,
        "completed_measured_optimizer_updates": len(rows),
        "amortized_samples_per_sec": total_samples / total_elapsed if total_elapsed else 0.0,
        "peak_allocated_gb": max((row["allocated_gb"] for row in rows), default=None),
        "peak_reserved_gb": max((row["reserved_gb"] for row in rows), default=None),
        "status": "accepted" if not reasons else "rejected",
        "rejection_reasons": sorted(set(reasons)),
    }
    return candidate_payload, rows, overhead, provenance


def _select(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["status"] == "accepted" and candidate["peak_reserved_gb"] is not None and candidate["peak_reserved_gb"] <= ELIGIBLE_RESERVED_GB
    ]
    if not eligible:
        rejected = sorted(
            {
                reason
                for candidate in candidates
                for reason in candidate.get("rejection_reasons", [])
            }
        )
        detail = "; ".join(rejected) if rejected else "no candidate completed the profile contract"
        raise NoEligibleRuntimeProfile(
            f"no eligible candidate with reserved<=42GB; refusing the 42-45GB range: {detail}"
        )
    highest = max(float(candidate["amortized_samples_per_sec"]) for candidate in eligible)
    near = [candidate for candidate in eligible if (highest - float(candidate["amortized_samples_per_sec"])) / highest < 0.03]
    selected = min(near, key=lambda item: (float(item["peak_reserved_gb"]), -float(item["amortized_samples_per_sec"]), item["name"]))
    if len(near) > 1:
        reason = "throughput is within 3% of the fastest eligible candidate; selected lower peak reserved memory"
    else:
        reason = "highest amortized throughput among candidates with reserved<=42GB"
    return selected, reason


def profile_runtime(
    *,
    runner_factory: Callable[..., RuntimeRunner],
    output_dir: str | Path,
    device: str = "cuda",
    clock: Callable[[], float] | None = None,
    memory_probe: MemoryProbe | None = None,
) -> dict[str, Any]:
    """Profile the three contractual candidates and atomically publish artifacts.

    CPU use is intentionally test-only: both a clock and a memory probe must be
    injected.  Production defaults use ``time.perf_counter`` and torch.cuda.
    """
    if not callable(runner_factory):
        raise TypeError("runner_factory must be callable")
    if device.startswith("cuda"):
        resolved_clock = clock or time.perf_counter
        resolved_memory: MemoryProbe = memory_probe or _CudaMemoryProbe()
    else:
        if clock is None or memory_probe is None:
            raise ValueError("CPU profiling requires explicit injectable clock and memory_probe")
        resolved_clock = clock
        resolved_memory = memory_probe
    root = Path(output_dir)
    candidate_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    overhead_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] | None = None
    for candidate in CANDIDATES:
        result, rows, overhead, candidate_provenance = _profile_candidate(
            runner_factory, candidate, device=device, clock=resolved_clock, memory_probe=resolved_memory
        )
        if provenance is None:
            provenance = candidate_provenance
        elif provenance != candidate_provenance:
            raise ValueError("all runtime candidates must share one P18 provenance identity")
        candidate_rows.append(result)
        step_rows.extend(rows)
        overhead_rows.append(overhead)
    assert provenance is not None
    runtime_payload = {
        **provenance,
        "profile_contract": {
            "candidates": [candidate.payload() for candidate in CANDIDATES],
            "warmup_optimizer_updates": WARMUP_OPTIMIZER_UPDATES,
            "measured_optimizer_updates": MEASURED_OPTIMIZER_UPDATES,
            "core_mechanism_flags": CORE_MECHANISM_FLAGS,
            "counterfactual_in_core_timing": False,
            "selection_reserved_limit_gb": ELIGIBLE_RESERVED_GB,
            "hard_rejection_reserved_limit_gb": MAX_RESERVED_GB,
        },
        "candidates": candidate_rows,
        "counterfactual_overhead": overhead_rows,
    }
    _validate_run_json("runtime_profile.json", runtime_payload)
    for row in step_rows:
        _validate_run_jsonl_row("runtime_steps.jsonl", row)
    try:
        selected, reason = _select(candidate_rows)
    except NoEligibleRuntimeProfile:
        _atomic_json(root / "runtime_profile.json", runtime_payload)
        _atomic_jsonl(root / "runtime_steps.jsonl", step_rows)
        raise
    selected_payload = {
        **provenance,
        "selected": selected,
        "reason": reason,
        "profile": {"core_mechanism_flags": CORE_MECHANISM_FLAGS, "counterfactual_in_core_timing": False},
    }
    _validate_run_json("selected_runtime_profile.json", selected_payload)
    _atomic_json(root / "runtime_profile.json", runtime_payload)
    _atomic_jsonl(root / "runtime_steps.jsonl", step_rows)
    _atomic_json(root / "selected_runtime_profile.json", selected_payload)
    return selected
