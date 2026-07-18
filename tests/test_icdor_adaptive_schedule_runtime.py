from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from fate_oia.engine.mosaic_icdor_adaptive_schedule import ICDORAdaptiveSchedule


def test_train_audit_references_survive_checkpoint_state_roundtrip() -> None:
    source = ICDORAdaptiveSchedule(pilot=True)
    source.record_train_audit_reference(
        joint=0.61, exp_map=0.42, entered_safe_joint=False
    )
    source.record_train_audit_reference(
        joint=0.59, exp_map=0.43, entered_safe_joint=True
    )
    payload = source.state_dict()

    restored = ICDORAdaptiveSchedule(pilot=True)
    restored.load_state_dict(payload)

    assert restored.best_train_audit_joint == pytest.approx(0.61)
    assert restored.safe_joint_entry_exp_map == pytest.approx(0.43)
from fate_oia.engine.train_acpr_mosaic_trust_icdor import (
    _adaptive_phase,
    _build_rank_queues,
    _load_resume,
    _save_checkpoint,
)
from fate_oia.optim.mosaic_action_pareto_admission import MOSAICActionParetoAdmission
from fate_oia.utils.mosaic_icdor_artifacts import (
    initialize_icdor_run_artifacts,
    write_icdor_adaptive_schedule_transition,
)


class _ResumeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(2, 2)
        self.action_router = nn.Module()
        self.action_router.register_buffer("edge_admission_mask", torch.zeros(2, 1, 1, dtype=torch.bool))

    def load_factor_certificate(self, payload: dict[str, object]) -> None:
        self.certificate = payload

    def set_edge_admission(self, mask: torch.Tensor) -> None:
        self.action_router.edge_admission_mask.copy_(mask)


def test_runtime_phase_comes_from_adaptive_state_not_epoch() -> None:
    schedule = ICDORAdaptiveSchedule(pilot=False)
    # CREDO grants shadow-learning access from FOUNDATION. Discrete evidence
    # admission only controls the final action route.
    assert _adaptive_phase(schedule).route_mode == "shadow"

    schedule.state = "DUAL_REASON_SHADOW"
    assert _adaptive_phase(schedule).route_mode == "shadow"

    schedule.state = "SAFE_JOINT"
    assert _adaptive_phase(schedule).route_mode == "admitted"

    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    main = source[source.index("def main"):]
    assert "get_icdor_phase(" not in main
    assert "get_icdor_pilot_phase(" not in main


def test_runtime_checkpoint_restores_adaptive_state_and_evidence_hashes(tmp_path: Path) -> None:
    model = _ResumeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    action_queue, reason_queue, observed_reason_queue = _build_rank_queues(capacity=8, device=torch.device("cpu"))
    pareto = MOSAICActionParetoAdmission()
    schedule = ICDORAdaptiveSchedule(pilot=True)
    schedule.state = "SAFE_JOINT"
    schedule.state_epochs = 3
    schedule.history.append({"epoch": 5, "state_after": "SAFE_JOINT"})
    path = tmp_path / "checkpoint.pth"
    (tmp_path / "certificate.json").write_text(json.dumps({"sha256": "CERT"}), encoding="utf-8")
    (tmp_path / "edge.json").write_text(json.dumps({"sha256": "EDGE", "entries": {}}), encoding="utf-8")

    _save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=5, best_joint=0.5,
        certificate_sha256="CERT", edge_admission_sha256="EDGE", config_sha256="CONFIG",
        split_sha256="SPLIT", action_queue=action_queue, reason_queue=reason_queue,
        observed_reason_queue=observed_reason_queue, pareto=pareto,
        adaptive_schedule=schedule,
    )
    schedule.state = "FOUNDATION"
    schedule.state_epochs = 0

    result = _load_resume(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        certificate_path=tmp_path / "certificate.json", edge_path=tmp_path / "edge.json",
        config_sha256="CONFIG", split_sha256="SPLIT", action_queue=action_queue,
        reason_queue=reason_queue, observed_reason_queue=observed_reason_queue,
        pareto=pareto, adaptive_schedule=schedule,
    )

    assert result == (6, 0.5, "CERT", "EDGE")
    assert schedule.state == "SAFE_JOINT"
    assert schedule.state_epochs == 3
    assert schedule.history[-1]["state_after"] == "SAFE_JOINT"


def test_runtime_artifact_requires_train_only_transition_provenance(tmp_path: Path) -> None:
    initialize_icdor_run_artifacts(
        tmp_path,
        manifest={"run": "adaptive"}, config={"training": {}}, source_manifest={"source": "x"},
        split_manifest={"split": "x"}, runtime_selection={"runtime": "x"},
        factor_certificate={"artifact": "factor_certificate", "status": "pending"},
        edge_admission={"artifact": "edge_admission", "status": "pending"},
    )
    row = {
        "epoch": 3,
        "state_before": "FOUNDATION",
        "state_after": "DUAL_REASON_SHADOW",
        "state_epochs_before": 3,
        "state_epochs_after": 0,
        "ready": True,
        "failed_closed": False,
        "readiness": {
            "train_core": {"source_split": "train_core", "finite": True},
            "train_audit": {"source_split": "train_audit", "factor_audit_complete": True},
            "train_calib": {"source_split": "train_calib", "finite": True},
        },
        "certificate_sha256": "CERT",
        "edge_admission_sha256": None,
    }
    path = write_icdor_adaptive_schedule_transition(tmp_path, row)

    persisted = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert persisted["state_after"] == "DUAL_REASON_SHADOW"
    assert persisted["certificate_sha256"] == "CERT"

    row["readiness"]["train_audit"]["source_split"] = "test"
    with pytest.raises(ValueError, match="test"):
        write_icdor_adaptive_schedule_transition(tmp_path, row)
