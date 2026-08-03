from pathlib import Path

import pytest
import torch

from fate_oia.utils.save_artifacts import (
    hash_named_tensors,
    save_epoch_artifacts,
    save_source_tree_hash,
    validate_epoch_artifacts,
)


def _kwargs():
    return {
        "metrics_raw": {"Act_mF1": 0.5},
        "metrics_deploy": {"Act_mF1": 0.5},
        "logits": {"action_final": torch.randn(2, 4), "reason_final": torch.randn(2, 21)},
        "labels": {"action": torch.zeros(2, 4), "reason": torch.zeros(2, 21)},
        "file_names": ["a.jpg", "b.jpg"],
        "mechanism": {"route": 1}, "utility": {"auc": 0.7},
        "faithfulness": {"gap": 0.1}, "gradient": {"clean": 1.0},
        "runtime": {"reserved_gb": 1.0},
        "git_head": "a" * 40,
        "config_hash": "config", "source_tree_hash": "source", "schema_hash": "schema",
        "split_manifest": {"test": [1, 2]}, "checkpoint": {"step": 1},
    }


def test_epoch_artifact_hash_chain_is_validated(tmp_path: Path):
    directory = save_epoch_artifacts(tmp_path, 1, **_kwargs())
    assert validate_epoch_artifacts(directory)
    (directory / "file_order.json").write_text('{"file_order":["b.jpg","a.jpg"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="file_order_hash mismatch"):
        validate_epoch_artifacts(directory)


def test_named_tensor_hash_is_binary_safe_and_content_sensitive():
    first = {"action": torch.tensor([[1.0, 2.0]])}
    same = {"action": torch.tensor([[1.0, 2.0]])}
    changed = {"action": torch.tensor([[1.0, 3.0]])}

    assert hash_named_tensors(first) == hash_named_tensors(same)
    assert hash_named_tensors(first) != hash_named_tensors(changed)


def test_epoch_artifact_requires_all_mechanism_groups(tmp_path: Path):
    kwargs = _kwargs()
    kwargs["utility"] = None
    with pytest.raises(ValueError, match="utility"):
        save_epoch_artifacts(tmp_path, 1, **kwargs)


def test_epoch_artifact_rejects_placeholder_binding_hashes(tmp_path: Path):
    kwargs = _kwargs()
    kwargs["git_head"] = "pending"
    directory = save_epoch_artifacts(tmp_path, 1, **kwargs)
    with pytest.raises(ValueError, match="placeholder hash"):
        validate_epoch_artifacts(directory)


def test_source_hash_binds_scripts_and_audit_skill(tmp_path: Path):
    (tmp_path / "fate_oia" / "models").mkdir(parents=True)
    (tmp_path / "configs").mkdir(); (tmp_path / "scripts").mkdir()
    skill = tmp_path / ".codex" / "skills" / "save-oia-implementation-audit"; skill.mkdir(parents=True)
    (tmp_path / "fate_oia" / "models" / "save_model.py").write_text("x=1", encoding="utf-8")
    script = tmp_path / "scripts" / "FATE_OIA_save_oia_v1_foreground.ps1"; script.write_text("one", encoding="utf-8")
    (skill / "SKILL.md").write_text("one", encoding="utf-8")
    first = save_source_tree_hash(tmp_path)
    script.write_text("two", encoding="utf-8")
    assert save_source_tree_hash(tmp_path) != first
