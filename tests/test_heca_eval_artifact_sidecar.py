from __future__ import annotations

from pathlib import Path

import torch
import pytest

from fate_oia.engine.eval_acpr_meter_oia import (
    CHEAP_SAME_FORWARD_MODES,
    INDEPENDENT_HECA_ABLATIONS,
    collect_outputs,
)
from fate_oia.utils.meter_artifacts import (
    validate_epoch_artifacts,
    validate_heca_artifact_sidecar,
    write_heca_artifact_sidecar,
)


class _CountingModel:
    """Small real interface double that exposes encode/decode call boundaries."""

    def __init__(self) -> None:
        self.encode_calls = 0
        self.decode_calls: list[tuple[str, ...]] = []

    def eval(self) -> "_CountingModel":
        return self

    def encode_images(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.encode_calls += 1
        return {"images": images}

    def decode_from_field(
        self,
        field: dict[str, torch.Tensor],
        *,
        diagnostic_modes: tuple[str, ...] = (),
        **_: object,
    ) -> dict[str, torch.Tensor]:
        self.decode_calls.append(diagnostic_modes)
        batch = field["images"].shape[0]
        offset = float(len(diagnostic_modes))
        return {
            "action_logits_visual": torch.zeros(batch, 4),
            "action_logits_final": torch.full((batch, 4), offset),
            "reason_logits_global": torch.zeros(batch, 21),
            "reason_logits_final": torch.full((batch, 21), offset),
            "factor_anchor_map": torch.full((batch, 21, 2), 0.5),
            "factor_null_mass": torch.full((batch, 21), 0.5),
            "factor_state_prob": torch.full((batch, 21, 3), 1.0 / 3.0),
            "factor_state_entropy": torch.ones(batch, 21),
            "factor_observability": torch.ones(batch, 21),
            "factor_reliability": torch.ones(batch, 21),
            "factor_layer_weights": torch.full((21, 3), 1.0 / 3.0),
            "action_evidence_delta": torch.zeros(batch, 4),
            "action_factor_weights": torch.full((batch, 4, 21), 1.0 / 21.0),
            "action_factor_contributions": torch.zeros(batch, 4, 21),
            "action_correction_rms_ratio": torch.zeros(batch, 4),
            "reason_evidence_delta": torch.zeros(batch, 21),
            "reason_groundable_mask": torch.ones(21),
        }


def _loader() -> list[dict[str, object]]:
    return [
        {
            "image": torch.zeros(2, 3, 8, 8),
            "action": torch.zeros(2, 4),
            "reason": torch.zeros(2, 21),
            "file_name": ["a.jpg", "b.jpg"],
        },
        {
            "image": torch.zeros(1, 3, 8, 8),
            "action": torch.zeros(1, 4),
            "reason": torch.zeros(1, 21),
            "file_name": ["c.jpg"],
        },
    ]


def test_cheap_diagnostics_share_one_dino_encode_per_batch() -> None:
    model = _CountingModel()
    collected = collect_outputs(
        model, _loader(), torch.device("cpu"), progress=1.0
    )

    assert model.encode_calls == 2
    assert model.decode_calls.count(()) == 2
    for mode in CHEAP_SAME_FORWARD_MODES.values():
        assert model.decode_calls.count(mode) == 2
    assert all(payload["dino_call_count"] == 0 for payload in collected["modes"].values())
    assert collected["dino_call_count"] == 2


def test_cheap_diagnostics_reject_hidden_extra_backbone_calls() -> None:
    model = _CountingModel()
    model.foundation = type("Foundation", (), {"ordinary_dino_calls": 0})()
    original_encode = model.encode_images

    def hidden_double_encode(images: torch.Tensor) -> dict[str, torch.Tensor]:
        field = original_encode(images)
        model.foundation.ordinary_dino_calls += 2
        return field

    model.encode_images = hidden_double_encode  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="DINO call mismatch"):
        collect_outputs(model, _loader(), torch.device("cpu"), progress=1.0)


def test_b0_b5_are_not_declared_as_same_forward_diagnostics() -> None:
    cheap = set(CHEAP_SAME_FORWARD_MODES)
    assert cheap.isdisjoint(INDEPENDENT_HECA_ABLATIONS)
    assert set(INDEPENDENT_HECA_ABLATIONS) == {f"B{index}" for index in range(6)}
    assert all(
        entry["execution"] == "independent_run"
        for entry in INDEPENDENT_HECA_ABLATIONS.values()
    )


def test_heca_sidecar_validator_requires_all_schema_artifacts(tmp_path: Path) -> None:
    payload = {
        "ontology_manifest": {"schema_version": 1, "factor_count": 21, "state_count": 3, "sha256": "a"},
        "tau_stats": {"source_split": "train_main", "alpha": 20.0, "tau": [0.5] * 21},
        "gradient_ownership": [{"optimizer_step": 1, "action_to_anchor_query": 0.0, "action_to_state_bridge_ratio": 0.05, "reason_to_action_credit": 0.0, "measurement_to_foundation": 0.0}],
        "loss_wiring": {"registry": ["action_final"], "counts": {"action_final": 1}, "duplicates": [], "pass": True},
        "component_call_counters": {"components": {"dino_encode": 1, "typed_measurement": 1, "action_credit": 1, "reason_correction": 1}, "one_dino_encode_per_batch": True},
        "contribution_conservation": [{"action": 0, "sum_contribution": 0.1, "action_credit_sum": 0.1, "abs_error": 0.0}],
        "schedule_state": {"optimizer_step": 1, "progress": 0.1, "credit_ramp": 0.5, "foundation_grad_cap": 0.25, "excess_risk": {"action": 0.0, "reason": 0.0}},
        "ablation_manifest": {"cheap_same_forward": list(CHEAP_SAME_FORWARD_MODES), "independent_runs": INDEPENDENT_HECA_ABLATIONS},
        "gates": {letter: {"gate": letter, "pass": True, "evidence": {"rows": 1}} for letter in "ABCDEFG"},
    }
    write_heca_artifact_sidecar(tmp_path, payload)
    assert validate_heca_artifact_sidecar(tmp_path) == []

    (tmp_path / "HECA_GATE_G.json").unlink()
    failures = validate_heca_artifact_sidecar(tmp_path)
    assert "HECA_GATE_G.json:missing" in failures


def test_epoch_validator_dispatches_to_heca_sidecar_validation(tmp_path: Path) -> None:
    payload = {
        "ontology_manifest": {"schema_version": 1, "factor_count": 21, "state_count": 3, "sha256": "a"},
        "tau_stats": {"source_split": "train_main", "alpha": 20.0, "tau": [0.5] * 21},
        "gradient_ownership": [{"optimizer_step": 1, "action_to_anchor_query": 0.0, "action_to_state_bridge_ratio": 0.05, "reason_to_action_credit": 0.0, "measurement_to_foundation": 0.0}],
        "loss_wiring": {"registry": ["action_final"], "counts": {"action_final": 1}, "duplicates": [], "pass": True},
        "component_call_counters": {"components": {"dino_encode": 1, "typed_measurement": 1, "action_credit": 1, "reason_correction": 1}, "one_dino_encode_per_batch": True},
        "contribution_conservation": [{"action": 0, "sum_contribution": 0.1, "action_credit_sum": 0.1, "abs_error": 0.0}],
        "schedule_state": {"optimizer_step": 1, "progress": 0.1, "credit_ramp": 0.5, "foundation_grad_cap": 0.25, "excess_risk": {"action": 0.0, "reason": 0.0}},
        "ablation_manifest": {"cheap_same_forward": list(CHEAP_SAME_FORWARD_MODES), "independent_runs": INDEPENDENT_HECA_ABLATIONS},
        "gates": {letter: {"gate": letter, "pass": True, "evidence": {"rows": 1}} for letter in "ABCDEFG"},
    }
    write_heca_artifact_sidecar(tmp_path, payload)
    (tmp_path / "HECA_GATE_A.json").unlink()

    assert "HECA_GATE_A.json:missing" in validate_epoch_artifacts(tmp_path)
