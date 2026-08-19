from pathlib import Path

import pytest
import torch
from torch import nn

from fate_oia.engine.train_vetra_staged_refine import (
    choose_refiner_candidate,
    freeze_base_model,
    make_stage_b_checkpoint,
    verify_reason_identity,
)


def test_freeze_base_model_disables_all_gradients_and_keeps_eval_mode():
    base = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
    freeze_base_model(base)
    assert not base.training
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_reason_identity_requires_exact_values_and_rejects_mutation():
    source = torch.randn(5, 21)
    verify_reason_identity(source, source)
    verify_reason_identity(source, source.clone())
    changed = source.clone()
    changed[0, 0] += 1e-6
    with pytest.raises(RuntimeError, match="reason identity"):
        verify_reason_identity(source, changed)


def test_candidate_selection_fails_closed_without_train_audit_gain():
    rows = [
        {"name": "base", "audit_mf1": 0.710, "audit_map": 0.790},
        {"name": "epoch_0", "audit_mf1": 0.711, "audit_map": 0.788},
        {"name": "epoch_1", "audit_mf1": 0.709, "audit_map": 0.795},
    ]
    selected = choose_refiner_candidate(rows, min_mf1_gain=0.002, max_map_drop=0.001)
    assert selected["name"] == "base"
    assert selected["refiner_selected"] is False


def test_candidate_selection_accepts_refiner_only_with_guarded_gain():
    rows = [
        {"name": "base", "audit_mf1": 0.710, "audit_map": 0.790},
        {"name": "epoch_0", "audit_mf1": 0.713, "audit_map": 0.7905},
        {"name": "epoch_1", "audit_mf1": 0.714, "audit_map": 0.787},
    ]
    selected = choose_refiner_candidate(rows, min_mf1_gain=0.002, max_map_drop=0.001)
    assert selected["name"] == "epoch_0"
    assert selected["refiner_selected"] is True


def test_stage_b_checkpoint_records_parent_and_explicit_base_fallback(tmp_path: Path):
    parent = tmp_path / "stage_a.pth"
    parent.write_bytes(b"same-run-stage-a")
    identity = {
        "run_id": "run",
        "run_root": str(tmp_path.resolve()),
        "git_head": "head",
        "source_tree_hash": "tree",
        "split_manifest_hash": "split",
    }
    payload = make_stage_b_checkpoint(
        parent_path=parent,
        identity=identity,
        refiner_selected=False,
        refiner_state=None,
        deployment_gain=torch.zeros(4),
        selection={"name": "base"},
    )
    assert payload["stage"] == "action_refined"
    assert payload["refiner_selected"] is False
    assert payload["refiner"] is None
    assert payload["run_identity"] == identity
    assert len(payload["parent_checkpoint_sha256"]) == 64
