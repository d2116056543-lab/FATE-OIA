from __future__ import annotations

import json

import pytest
import torch

from fate_oia.engine.train_acpr_mosaic_ad import _build_failure_cases
from fate_oia.utils.mosaic_artifacts import (
    EPOCH_JSONL_FILES,
    EPOCH_JSON_FILES,
    LOGIT_FILES,
    initialize_run_artifacts,
    validate_artifact_schema,
    write_epoch_artifacts,
)


def _root(tmp_path):
    return initialize_run_artifacts(
        tmp_path,
        manifest={"direct_image": True},
        config={"training": {"epochs": 15}},
        git_state={"head": "abc"},
        runtime_profile={"selected": {"batch_size": 4}},
        split_stats={"split_hash": "def"},
    )


def test_full_epoch_artifact_schema_round_trip(tmp_path) -> None:
    root = _root(tmp_path)
    write_epoch_artifacts(
        root,
        epoch=0,
        json_payloads={name: {"epoch": 0, "artifact": name} for name in EPOCH_JSON_FILES},
        jsonl_payloads={name: [{"epoch": 0, "artifact": name}] for name in EPOCH_JSONL_FILES},
        logits={name: torch.zeros(2, 4 if "action" in name else 21) for name in LOGIT_FILES},
        sample_ids=["a.jpg", "b.jpg"],
    )
    result = validate_artifact_schema(root, epochs=[0])
    assert result == {"pass": True, "missing": [], "invalid": []}
    assert json.loads((root / "epoch_000" / "metrics_summary.json").read_text())['epoch'] == 0


def test_artifact_writer_fails_closed_on_missing_schema_member(tmp_path) -> None:
    root = _root(tmp_path)
    with pytest.raises(ValueError):
        write_epoch_artifacts(
            root,
            epoch=0,
            json_payloads={},
            jsonl_payloads={},
            logits={},
            sample_ids=[],
        )


def test_artifact_validator_reports_removed_tensor(tmp_path) -> None:
    root = _root(tmp_path)
    write_epoch_artifacts(
        root,
        epoch=1,
        json_payloads={name: {"epoch": 1} for name in EPOCH_JSON_FILES},
        jsonl_payloads={name: [{"epoch": 1}] for name in EPOCH_JSONL_FILES},
        logits={name: torch.ones(1, 4 if "action" in name else 21) for name in LOGIT_FILES},
        sample_ids=["sample.jpg"],
    )
    (root / "epoch_001" / "logits" / "reason_deploy.pt").unlink()
    result = validate_artifact_schema(root, epochs=[1])
    assert not result["pass"]
    assert any(path.endswith("reason_deploy.pt") for path in result["missing"])


def test_failure_cases_are_derived_from_real_prediction_errors() -> None:
    logits_action = torch.tensor([[4.0, -4.0, 4.0, -4.0], [-4.0, 4.0, -4.0, 4.0]])
    labels_action = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    logits_reason = torch.full((2, 21), -4.0)
    labels_reason = torch.zeros(2, 21)
    labels_reason[1, 5] = 1.0

    rows = _build_failure_cases(
        logits_action,
        logits_reason,
        labels_action,
        labels_reason,
        ["sample-a", "sample-b"],
        epoch=3,
    )

    assert rows
    assert rows[0]["available"] is True
    assert rows[0]["file_name"] in {"sample-a", "sample-b"}
    assert rows[0]["action_error_count"] + rows[0]["reason_error_count"] > 0


def test_strict_artifact_validation_rejects_placeholder_payloads(tmp_path) -> None:
    root = _root(tmp_path)
    write_epoch_artifacts(
        root,
        epoch=0,
        json_payloads={name: {"epoch": 0, "artifact": name} for name in EPOCH_JSON_FILES},
        jsonl_payloads={name: [{"epoch": 0, "available": False}] for name in EPOCH_JSONL_FILES},
        logits={name: torch.zeros(2, 4 if "action" in name else 21) for name in LOGIT_FILES},
        sample_ids=["a.jpg", "b.jpg"],
    )
    result = validate_artifact_schema(root, epochs=[0], strict_semantics=True)
    assert not result["pass"]
    assert result["semantic_errors"]
