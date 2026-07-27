"""RED contracts for the P20 RAEL runtime profiler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import fate_oia.engine.profile_acpr_rael_oia as profile_acpr_rael_oia
import fate_oia.utils.rael_artifacts as rael_artifacts
import fate_oia.utils.rael_runtime as rael_runtime

def _provenance() -> dict[str, object]:
    return {
        "schema_version": "rael-artifact-v1",
        "producer": "fate_oia.engine.train_acpr_rael_oia:runtime-profile",
        "source_fingerprint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


class _MemoryProbe:
    def __init__(self, *, allocated: float = 6.0, reserved: float = 12.0) -> None:
        self.allocated = allocated
        self.reserved = reserved
        self.before_calls = 0
        self.after_calls = 0

    def before_update(self, device: str) -> None:
        assert device == "cpu"
        self.before_calls += 1

    def after_update(self, device: str) -> tuple[float, float]:
        assert device == "cpu"
        self.after_calls += 1
        return self.allocated, self.reserved


class _Runner:
    def __init__(
        self,
        candidate: dict[str, int | str],
        *,
        reserved: float = 12.0,
        finite: bool = True,
        sample_delta: int = 0,
        microbatches: int | None = None,
        dino_calls_per_microbatch: int = 1,
        dino_call_count_total: int | None = None,
        flags: dict[str, bool] | None = None,
        counterfactual_overrides: dict[str, object] | None = None,
        oom: bool = False,
    ) -> None:
        self.candidate = candidate
        self.reserved = reserved
        self.finite = finite
        self.sample_delta = sample_delta
        self.microbatches = microbatches
        self.dino_calls_per_microbatch = dino_calls_per_microbatch
        self.dino_call_count_total = dino_call_count_total
        self.flags = flags or dict(rael_runtime.CORE_MECHANISM_FLAGS)
        self.counterfactual_overrides = counterfactual_overrides or {}
        self.oom = oom
        self.configurations: list[dict[str, object]] = []
        self.update_contexts: list[dict[str, object]] = []
        self.counterfactual_calls = 0

    def artifact_provenance(self) -> dict[str, object]:
        return _provenance()

    def configure_runtime_profile(self, context: dict[str, object]) -> None:
        self.configurations.append(context)

    def optimizer_update(self, context: dict[str, object]) -> dict[str, object]:
        self.update_contexts.append(context)
        if self.oom:
            raise RuntimeError("CUDA out of memory")
        gradient_accumulation_steps = int(self.candidate["gradient_accumulation_steps"])
        return {
            "samples": int(self.candidate["batch_size"]) * gradient_accumulation_steps + self.sample_delta,
            "microbatches": self.microbatches if self.microbatches is not None else gradient_accumulation_steps,
            "finite": self.finite,
            "dino_call_count_per_microbatch": self.dino_calls_per_microbatch,
            "dino_call_count_total": self.dino_call_count_total if self.dino_call_count_total is not None else gradient_accumulation_steps,
            "mechanism_flags": self.flags,
            "data_time": 0.01,
            "dino_time": 0.02,
        }

    def measure_counterfactual_overhead(self, context: dict[str, object]) -> dict[str, object]:
        self.counterfactual_calls += 1
        assert context["counterfactual"] is True
        return {
            "samples": int(self.candidate["batch_size"]),
            "counterfactual_executed": True,
            "available": True,
            "valid_target_count": 1,
            "dino_call_count": 0,
            "finite": True,
            "loss": 0.1,
            **self.counterfactual_overrides,
        }


def _factory_with(runners: dict[str, _Runner]):
    def factory(*, candidate: dict[str, int | str], device: str) -> _Runner:
        assert device == "cpu"
        return runners[str(candidate["name"])]

    return factory


def _profile(tmp_path: Path, runners: dict[str, _Runner], *, memory: _MemoryProbe | None = None):
    return rael_runtime.profile_runtime(
        runner_factory=_factory_with(runners),
        output_dir=tmp_path,
        device="cpu",
        clock=_Clock(),
        memory_probe=memory or _MemoryProbe(),
    )


def _runners(**kwargs: object) -> dict[str, _Runner]:
    candidates = (("P0", 8, 4, 8), ("P1", 6, 5, 8), ("P2", 4, 8, 8))
    return {
        name: _Runner(
            {"name": name, "batch_size": batch, "gradient_accumulation_steps": accum, "num_workers": workers},
            **kwargs,
        )
        for name, batch, accum, workers in candidates
    }


def test_each_fixed_candidate_uses_exact_warmup_measurement_and_accumulation(tmp_path: Path) -> None:
    runners = _runners()
    selected = _profile(tmp_path, runners)

    assert selected["name"] == "P0"
    for runner in runners.values():
        assert len(runner.update_contexts) == 25
        assert [item["phase"] for item in runner.update_contexts].count("warmup") == 5
        assert [item["phase"] for item in runner.update_contexts].count("measured") == 20
        assert all(item["gradient_accumulation_steps"] == runner.candidate["gradient_accumulation_steps"] for item in runner.update_contexts)
        assert runner.counterfactual_calls == 1


def test_near_throughput_tie_prefers_lower_peak_reserved_memory(tmp_path: Path) -> None:
    runners = _runners()
    memory = _MemoryProbe(reserved=12.0)
    selected = _profile(tmp_path, runners, memory=memory)
    assert selected["name"] == "P0"

    selected, _ = rael_runtime._select(
        [
            {"name": "P0", "status": "accepted", "amortized_samples_per_sec": 100.0, "peak_reserved_gb": 12.0},
            {"name": "P1", "status": "accepted", "amortized_samples_per_sec": 98.0, "peak_reserved_gb": 10.0},
            {"name": "P2", "status": "accepted", "amortized_samples_per_sec": 70.0, "peak_reserved_gb": 8.0},
        ]
    )
    assert selected["name"] == "P1"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"oom": True}, "oom"),
        ({"finite": False}, "finite"),
        ({"sample_delta": -1}, "samples"),
        ({"microbatches": 1}, "gradient_accumulation_steps"),
        ({"dino_calls_per_microbatch": 2}, "dino_call_count_per_microbatch"),
        ({"dino_call_count_total": 1}, "dino_call_count_total"),
        ({"flags": {"dino": True, "four_layers": True, "slot": False, "unary": True, "pairwise": True, "gradient_admission": True, "mirror_25pct": True, "counterfactual": False}}, "mechanism"),
        ({"flags": {"dino": True, "four_layers": True, "slot": True, "unary": True, "pairwise": True, "gradient_admission": True, "mirror_25pct": True}}, "mechanism"),
        ({"flags": {**rael_runtime.CORE_MECHANISM_FLAGS, "unexpected": True}}, "mechanism"),
    ],
)
def test_rejects_oom_nonfinite_dino_multiplicity_and_disabled_mechanisms(tmp_path: Path, kwargs: dict[str, object], reason: str) -> None:
    runners = _runners(**kwargs)
    with pytest.raises(rael_runtime.NoEligibleRuntimeProfile, match=reason):
        _profile(tmp_path, runners)
    artifact = json.loads((tmp_path / "runtime_profile.json").read_text(encoding="utf-8"))
    assert all(any(reason in detail for detail in candidate["rejection_reasons"]) for candidate in artifact["candidates"])


@pytest.mark.parametrize("reserved, expects_selection", [(42.0, True), (42.0001, False), (45.0, False)])
def test_memory_boundaries_fail_closed(tmp_path: Path, reserved: float, expects_selection: bool) -> None:
    runners = _runners()
    if expects_selection:
        assert _profile(tmp_path, runners, memory=_MemoryProbe(reserved=reserved))["name"] == "P0"
    else:
        with pytest.raises(rael_runtime.NoEligibleRuntimeProfile, match="reserved"):
            _profile(tmp_path, runners, memory=_MemoryProbe(reserved=reserved))
        assert not (tmp_path / "selected_runtime_profile.json").exists()


def test_counterfactual_overhead_is_measured_outside_core_updates(tmp_path: Path) -> None:
    runners = _runners()
    _profile(tmp_path, runners)
    payload = json.loads((tmp_path / "runtime_profile.json").read_text(encoding="utf-8"))
    assert len(payload["counterfactual_overhead"]) == 3
    for item in payload["counterfactual_overhead"]:
        assert item["samples"] > 0 and item["elapsed"] > 0
        assert item["counterfactual_executed"] is True
        assert item["available"] is True
        assert item["valid_target_count"] > 0
        assert item["dino_call_count"] == 0
        assert item["finite"] is True
        assert item["loss"] == pytest.approx(0.1)
    assert all(not context["counterfactual"] for runner in runners.values() for context in runner.update_contexts)


def test_initial_no_eligible_control_is_recorded_without_faking_a_loss(
    tmp_path: Path,
) -> None:
    runners = _runners(
        counterfactual_overrides={
            "available": False,
            "reason": "no_eligible_control",
            "valid_target_count": 0,
            "loss": None,
        }
    )
    assert _profile(tmp_path, runners)["name"] == "P0"
    payload = json.loads(
        (tmp_path / "runtime_profile.json").read_text(encoding="utf-8")
    )
    for item in payload["counterfactual_overhead"]:
        assert item["counterfactual_executed"] is True
        assert item["available"] is False
        assert item["reason"] == "no_eligible_control"
        assert item["valid_target_count"] == 0
        assert item["loss"] is None


@pytest.mark.parametrize(
    ("counterfactual_overrides", "reason"),
    [
        ({"counterfactual_executed": False}, "counterfactual_executed"),
        ({"available": False}, "available"),
        ({"valid_target_count": 0}, "valid_target_count"),
        ({"dino_call_count": 1}, "dino_call_count"),
        ({"finite": False}, "finite"),
        ({"loss": float("nan")}, "loss"),
    ],
)
def test_rejects_candidate_when_counterfactual_replay_is_not_real(
    tmp_path: Path,
    counterfactual_overrides: dict[str, object],
    reason: str,
) -> None:
    with pytest.raises(rael_runtime.NoEligibleRuntimeProfile, match=reason):
        _profile(tmp_path, _runners(counterfactual_overrides=counterfactual_overrides))
    artifact = json.loads((tmp_path / "runtime_profile.json").read_text(encoding="utf-8"))
    assert all(any(reason in detail for detail in candidate["rejection_reasons"]) for candidate in artifact["candidates"])


def test_artifacts_have_p18_provenance_and_per_update_schema(tmp_path: Path) -> None:
    _profile(tmp_path, _runners())
    runtime = json.loads((tmp_path / "runtime_profile.json").read_text(encoding="utf-8"))
    selected = json.loads((tmp_path / "selected_runtime_profile.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (tmp_path / "runtime_steps.jsonl").read_text(encoding="utf-8").splitlines()]

    for payload in (runtime, selected, *rows):
        assert payload["schema_version"] == "rael-artifact-v1"
        assert payload["producer"] == _provenance()["producer"]
        assert len(payload["source_fingerprint_sha256"]) == 64
        assert len(payload["config_sha256"]) == 64
    assert len(rows) == 60
    assert {"elapsed", "samples_per_sec", "allocated_gb", "reserved_gb", "dino_call_count", "dino_call_count_total", "finite", "mechanism_flags"}.issubset(rows[0])
    assert rows[0]["dino_call_count"] == 1
    assert rows[0]["dino_call_count_total"] == rows[0]["microbatches"] == 4
    assert set(rows[0]["mechanism_flags"]) == {*rael_runtime.CORE_MECHANISM_FLAGS, "ledger"}
    assert selected["selected"]["effective_batch_size"] == 32
    assert selected["reason"]


def test_publication_invokes_p18_validators_for_every_runtime_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_calls: list[str] = []
    jsonl_calls: list[str] = []

    def validate_json(name: str, payload: object) -> dict[str, object]:
        json_calls.append(name)
        return rael_artifacts._validate_run_json(name, payload)

    def validate_jsonl(name: str, row: object) -> dict[str, object]:
        jsonl_calls.append(name)
        return rael_artifacts._validate_run_jsonl_row(name, row)

    monkeypatch.setattr(rael_runtime, "_validate_run_json", validate_json)
    monkeypatch.setattr(rael_runtime, "_validate_run_jsonl_row", validate_jsonl)
    _profile(tmp_path, _runners())

    assert json_calls == ["runtime_profile.json", "selected_runtime_profile.json"]
    assert jsonl_calls == ["runtime_steps.jsonl"] * 60


def test_cli_requires_runner_factory() -> None:
    with pytest.raises(SystemExit) as error:
        profile_acpr_rael_oia.main(["--output_dir", "unused"])
    assert error.value.code == 2
