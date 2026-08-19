import json
from pathlib import Path

import pytest
import torch

from fate_oia.utils.vetra_stage_contracts import (
    atomic_write_json,
    build_run_identity,
    promote_stage_a_checkpoint,
    sha256_file,
    validate_stage_checkpoint,
)


def _base_checkpoint(identity: dict) -> dict:
    return {
        "model": {"weight": torch.ones(1)},
        "manifest": {
            "git_head": identity["git_head"],
            "source_tree_hash": identity["source_tree_hash"],
            "split_manifest_hash": identity["split_manifest_hash"],
            "external_task_checkpoint": None,
        },
        "selection_split": "train_audit",
        "selected_view": "ema",
        "epoch": 7,
    }


def test_promoted_stage_a_checkpoint_carries_clean_same_run_lineage(tmp_path: Path):
    split = tmp_path / "split_manifest.json"
    split.write_text('{"train": ["a.jpg"]}', encoding="utf-8")
    identity = build_run_identity(
        run_root=tmp_path,
        run_id="run-001",
        git_head="abc123",
        source_tree_hash="tree123",
        split_manifest_path=split,
    )
    source = tmp_path / "stage_a" / "checkpoint_final_train_audit_selected.pth"
    source.parent.mkdir()
    torch.save(_base_checkpoint(identity), source)

    promoted = tmp_path / "checkpoint_stage_a_selected.pth"
    metadata = promote_stage_a_checkpoint(source, promoted, identity)

    payload = torch.load(promoted, map_location="cpu", weights_only=False)
    assert payload["stage"] == "base_selected"
    assert payload["run_identity"] == identity
    assert payload["source_checkpoint_sha256"] == sha256_file(source)
    assert metadata["checkpoint_sha256"] == sha256_file(promoted)
    validate_stage_checkpoint(promoted, identity, expected_stage="base_selected")


def test_stage_contract_rejects_historical_or_cross_run_checkpoint(tmp_path: Path):
    split = tmp_path / "split_manifest.json"
    split.write_text("{}", encoding="utf-8")
    identity = build_run_identity(tmp_path, "run-current", "head", "tree", split)
    checkpoint = tmp_path / "historical.pth"
    torch.save({
        "stage": "base_selected",
        "run_identity": {**identity, "run_id": "run-historical"},
        "manifest": {"external_task_checkpoint": "old-task.pth"},
    }, checkpoint)

    with pytest.raises(RuntimeError, match="run_id|external task checkpoint"):
        validate_stage_checkpoint(checkpoint, identity, expected_stage="base_selected")


def test_stage_contract_rejects_parent_stage_or_split_mismatch(tmp_path: Path):
    split = tmp_path / "split_manifest.json"
    split.write_text("{}", encoding="utf-8")
    identity = build_run_identity(tmp_path, "run", "head", "tree", split)
    checkpoint = tmp_path / "stage_b.pth"
    torch.save({
        "stage": "action_refined",
        "run_identity": identity,
        "parent_checkpoint_sha256": "wrong-parent",
        "manifest": {"external_task_checkpoint": None},
    }, checkpoint)

    with pytest.raises(RuntimeError, match="parent"):
        validate_stage_checkpoint(
            checkpoint,
            identity,
            expected_stage="action_refined",
            expected_parent_sha256="expected-parent",
        )
    with pytest.raises(RuntimeError, match="stage"):
        validate_stage_checkpoint(checkpoint, identity, expected_stage="base_selected")

    changed = dict(identity)
    changed["split_manifest_hash"] = "different"
    with pytest.raises(RuntimeError, match="split_manifest_hash"):
        validate_stage_checkpoint(checkpoint, changed, expected_stage="action_refined")


def test_atomic_json_record_is_complete_and_parseable(tmp_path: Path):
    path = tmp_path / "stage_a_complete.json"
    atomic_write_json(path, {"complete": True, "stage": "stage_a"})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "complete": True,
        "stage": "stage_a",
    }
    assert not path.with_suffix(path.suffix + ".tmp").exists()
