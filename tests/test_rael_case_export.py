from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import Tensor

from fate_oia.engine.export_rael_cases import RAELCaseExportCollector
from fate_oia.models.rael_oia_model import BRANCH_NAMES
from fate_oia.utils.rael_artifacts import _validate_epoch_jsonl
from fate_oia.utils.rael_posthoc_calibration import fit_posthoc_calibration


ACTION_NAMES = ("forward", "stop", "left", "right")
REASON_NAMES = tuple(f"Driving reason {index + 1}" for index in range(21))
GRID_HW = (45, 80)
SLOTS = 20
PAIRS = SLOTS * (SLOTS - 1) // 2


def _calibration(*, targets: int) -> dict[str, Any]:
    logits = torch.linspace(-1.2, 1.3, 8 * targets, dtype=torch.float32).reshape(8, targets)
    labels = torch.zeros((8, targets), dtype=torch.float32)
    labels[::2, 0] = 1.0
    labels[1::3, min(1, targets - 1)] = 1.0
    return fit_posthoc_calibration(
        raw_logits=logits,
        labels=labels,
        split="train_calib",
        group_ids=[f"label-group-{index}" for index in range(targets)],
        stable_ids=[f"E:/bdd-oia/train_calib/calib-{index:04d}.jpg" for index in range(8)],
    )


def _batch_names(batch_size: int = 2) -> list[str]:
    return [f"E:/bdd-oia/test/case-{index:04d}.jpg" for index in range(batch_size)]


def _incident(pairwise: Tensor, pair_indices: Tensor) -> Tensor:
    result = pairwise.new_zeros(pairwise.shape[0], pairwise.shape[1], SLOTS)
    for side in (0, 1):
        result.scatter_add_(2, pair_indices[:, side].view(1, 1, -1).expand_as(pairwise), pairwise)
    return result


def _formal_outputs(batch_size: int = 2) -> dict[str, Any]:
    action_final = torch.tensor([[1.2, -0.9, 0.35, -0.2], [-0.7, 0.6, -0.4, 0.8]], dtype=torch.float32)
    reason_final = torch.linspace(-0.7, 0.9, batch_size * 21, dtype=torch.float32).reshape(batch_size, 21)
    branches: dict[str, dict[str, Tensor]] = {}
    for index, name in enumerate(BRANCH_NAMES):
        delta = (index + 1) * 0.001
        branches[name] = {"action": action_final.clone() if name == "full" else action_final + delta,
                          "reason": reason_final.clone() if name == "full" else reason_final - delta}

    pair_indices = torch.triu_indices(SLOTS, SLOTS, offset=1).transpose(0, 1).contiguous()
    action_unary = 0.002 + torch.arange(SLOTS, dtype=torch.float32).view(1, 1, SLOTS).expand(batch_size, 4, SLOTS) * 0.0001
    reason_unary = 0.001 + torch.arange(SLOTS, dtype=torch.float32).view(1, 1, SLOTS).expand(batch_size, 21, SLOTS) * 0.0001
    action_pairwise = 0.00001 + torch.arange(PAIRS, dtype=torch.float32).view(1, 1, PAIRS).expand(batch_size, 4, PAIRS) * 0.0000001
    reason_pairwise = 0.00002 + torch.arange(PAIRS, dtype=torch.float32).view(1, 1, PAIRS).expand(batch_size, 21, PAIRS) * 0.0000001
    action_incident = _incident(action_pairwise, pair_indices)
    reason_incident = _incident(reason_pairwise, pair_indices)

    masks = torch.linspace(0.001, 1.0, batch_size * SLOTS * GRID_HW[0] * GRID_HW[1], dtype=torch.float32).reshape(batch_size, SLOTS, *GRID_HW)
    return {
        "action_logits_final": action_final,
        "reason_logits_final": reason_final,
        "branch_logits": branches,
        "slot_masks": masks,
        "slot_area": masks.mean(dim=(-2, -1)),
        "slot_centroid": torch.stack([torch.full((batch_size, SLOTS), 0.4), torch.full((batch_size, SLOTS), 0.6)], dim=-1),
        "slot_scale": torch.full((batch_size, SLOTS), 0.2),
        "slot_activity": torch.full((batch_size, SLOTS), 0.9),
        "slot_presence": torch.full((batch_size, SLOTS), 0.8),
        "slot_observability": torch.full((batch_size, SLOTS), 0.75),
        "slot_reliability": torch.full((batch_size, SLOTS), 0.7),
        "slot_type_probs": torch.full((batch_size, 12, 6), 1.0 / 6.0),
        "slot_state_probs": torch.full((batch_size, 12, 4), 0.25),
        "slot_sector_probs": {"horizontal": torch.full((batch_size, SLOTS, 3), 1.0 / 3.0), "depth": torch.full((batch_size, SLOTS, 3), 1.0 / 3.0)},
        "action_global_contribution": action_final - action_unary.sum(dim=-1) - action_pairwise.sum(dim=-1),
        "reason_global_contribution": reason_final - reason_unary.sum(dim=-1) - reason_pairwise.sum(dim=-1),
        "action_unary_contributions": action_unary,
        "reason_unary_contributions": reason_unary,
        "action_pairwise_contributions": action_pairwise,
        "reason_pairwise_contributions": reason_pairwise,
        "action_pairwise_incident_contributions": action_incident,
        "reason_pairwise_incident_contributions": reason_incident,
        "action_analytical_deletion": action_unary + action_incident,
        "reason_analytical_deletion": reason_unary + reason_incident,
        "action_pair_indices": pair_indices,
        "reason_pair_indices": pair_indices.clone(),
    }


def _targets(batch_size: int = 2) -> tuple[Tensor, Tensor]:
    action = torch.zeros((batch_size, 4), dtype=torch.float32)
    action[0, 0] = 1.0
    action[1, 1] = 1.0
    reason = torch.zeros((batch_size, 21), dtype=torch.float32)
    reason[0, 0] = 1.0
    reason[1, 2] = 1.0
    return action, reason


def _provenance(*, sample_count: int = 2) -> dict[str, Any]:
    return {"schema_version": "rael-artifact-v1", "producer": "fate_oia.engine.export_rael_cases:p19-case-export", "source_fingerprint_sha256": "a" * 64, "config_sha256": "b" * 64, "epoch": 3, "sample_count": sample_count}


def _consume(collector: RAELCaseExportCollector, outputs: dict[str, Any] | None = None, *, names: list[str] | None = None) -> None:
    action, reason = _targets()
    collector.consume(file_names=_batch_names() if names is None else names, action_targets=action, reason_targets=reason,
                      outputs=_formal_outputs() if outputs is None else outputs, action_calibration=_calibration(targets=4),
                      reason_calibration=_calibration(targets=21), action_names=ACTION_NAMES, reason_names=REASON_NAMES)


def _collector(**kwargs: Any) -> RAELCaseExportCollector:
    return RAELCaseExportCollector(max_failure_cases=kwargs.pop("max_failure_cases", 3), max_evidence_cases=kwargs.pop("max_evidence_cases", 4), top_slots=kwargs.pop("top_slots", 2), **kwargs)


def test_p19_case_export_happy_path_uses_real_model_fields_and_validates_p18() -> None:
    first = _collector()
    _consume(first)
    rows = first.finalize(_provenance())
    assert set(rows) == {"failure_cases.jsonl", "evidence_cases.jsonl"}
    assert 0 < len(rows["failure_cases.jsonl"]) <= 3
    assert 0 < len(rows["evidence_cases.jsonl"]) <= 4
    for evidence in rows["evidence_cases.jsonl"]:
        data = evidence["data"]
        assert len(data["contributions"]["pairwise"]) == PAIRS
        assert data["contributions"]["pair_indices"] == torch.triu_indices(SLOTS, SLOTS, offset=1).transpose(0, 1).tolist()
        assert abs(data["contributions"]["reconstruction_error"]) <= 1e-5
        attributes = data["attributes"][str(data["selected_slots"][0])]
        assert "type" not in attributes and "state" not in attributes
        assert attributes["type_probabilities"] is None or len(attributes["type_probabilities"]) == 6
        assert attributes["state_probabilities"] is None or len(attributes["state_probabilities"]) == 4
        assert set(attributes["sector_probabilities"]) == {"horizontal", "depth"}
    _validate_epoch_jsonl("failure_cases.jsonl", rows["failure_cases.jsonl"], epoch=3)
    _validate_epoch_jsonl("evidence_cases.jsonl", rows["evidence_cases.jsonl"], epoch=3)
    second = _collector()
    _consume(second)
    assert second.finalize(_provenance()) == rows


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda output: output.pop("slot_area"), "slot_area"),
    (lambda output: output.__setitem__("slot_type_probs", torch.zeros((2, 12, 6))), "sum to one"),
    (lambda output: output["slot_masks"].fill_(float("nan")), "finite"),
    (lambda output: output["action_pairwise_incident_contributions"].add_(0.5), "incident"),
    (lambda output: output["action_global_contribution"].add_(0.5), "reconstruct"),
    (lambda output: output.__setitem__("reason_pair_indices", output["reason_pair_indices"].roll(1, dims=0)), "unordered pair"),
])
def test_p19_case_export_rejects_malformed_or_fabricated_real_api_fields(mutation: Any, message: str) -> None:
    output = _formal_outputs()
    mutation(output)
    with pytest.raises((TypeError, ValueError), match=message):
        _consume(_collector(), output)


def test_p19_case_export_records_zero_deletion_as_unavailable() -> None:
    output = _formal_outputs()
    output["action_unary_contributions"].zero_()
    output["action_pairwise_contributions"].zero_()
    output["action_pairwise_incident_contributions"].zero_()
    output["action_analytical_deletion"].zero_()
    output["action_global_contribution"] = output["action_logits_final"].clone()
    collector = _collector()
    _consume(collector, output)
    rows = collector.finalize(_provenance())
    evidence = rows["evidence_cases.jsonl"][0]["data"]
    assert evidence["available"] is False
    assert evidence["unavailable_reason"] == "zero_analytical_deletion"
    assert evidence["target"]["type"] == "action"
    _validate_epoch_jsonl("evidence_cases.jsonl", rows["evidence_cases.jsonl"], epoch=3)


def test_p19_case_export_requires_all_unordered_pairs_and_matching_action_reason_indices() -> None:
    output = _formal_outputs()
    output["reason_pair_indices"][0] = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="unordered pair"):
        _consume(_collector(), output)
    output = _formal_outputs()
    output["reason_pair_indices"] = output["reason_pair_indices"].clone()
    output["reason_pair_indices"][0] = torch.tensor([0, 2])
    with pytest.raises(ValueError, match="unordered pair"):
        _consume(_collector(), output)


def test_p19_case_export_rejects_bad_names_targets_and_calibration_leakage() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _consume(_collector(), names=[_batch_names()[0], _batch_names()[0]])
    action, reason = _targets()
    action[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        _collector().consume(file_names=_batch_names(), action_targets=action, reason_targets=reason, outputs=_formal_outputs(), action_calibration=_calibration(targets=4), reason_calibration=_calibration(targets=21), action_names=ACTION_NAMES, reason_names=REASON_NAMES)
    leaky = _calibration(targets=4)
    leaky["fit_split"] = "test"
    with pytest.raises(ValueError, match="train_calib"):
        _collector().consume(file_names=_batch_names(), action_targets=_targets()[0], reason_targets=_targets()[1], outputs=_formal_outputs(), action_calibration=leaky, reason_calibration=_calibration(targets=21), action_names=ACTION_NAMES, reason_names=REASON_NAMES)


def test_p19_case_export_is_bounded_immutable_and_never_keeps_gpu_tensors() -> None:
    collector = _collector(max_failure_cases=1, max_evidence_cases=1, top_slots=1)
    _consume(collector)
    rows = collector.finalize(_provenance())
    assert len(rows["failure_cases.jsonl"]) == len(rows["evidence_cases.jsonl"]) == 1
    with pytest.raises(RuntimeError, match="finalized"):
        _consume(collector)
    if torch.cuda.is_available():
        output = _formal_outputs()
        for key, value in list(output.items()):
            if isinstance(value, Tensor):
                output[key] = value.cuda()
        output["slot_sector_probs"] = {key: value.cuda() for key, value in output["slot_sector_probs"].items()}
        output["branch_logits"] = {name: {key: value.cuda() for key, value in branch.items()} for name, branch in output["branch_logits"].items()}
        gpu_collector = _collector()
        _consume(gpu_collector, output)
        assert all(not isinstance(value, Tensor) for value in gpu_collector._failure_cases + gpu_collector._evidence_cases)
