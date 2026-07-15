from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch

from fate_oia.utils.mosaic_icdor_artifacts import (
    ICDOR_EPOCH_JSON_FILES,
    ICDOR_EPOCH_JSONL_FILES,
    ICDOR_LOGIT_FILES,
    initialize_icdor_run_artifacts,
    validate_icdor_artifact_schema,
    write_icdor_epoch_artifacts,
    _matched_control_provenance_valid,
)


def _matched_arms() -> list[dict[str, object]]:
    common = {
        "available_sample_count": 2,
        "max_mass_error": 0.0,
        "max_overlap": 0.0,
    }
    return [
        {
            **common,
            "control_type": "same_type_identity",
            "identity_source_factor_names": ["identity_factor"],
            "identity_source_factor_types": ["object"],
            "identity_source_regions": ["front"],
            "factor_type": "object",
            "region": "front",
            "spatial_offsets": [],
        },
        *[
            {
                **common,
                "control_type": "spatial_roll",
                "identity_source_factor_names": [],
                "identity_source_factor_types": [],
                "identity_source_regions": [],
                "factor_type": "object",
                "region": "front",
                "spatial_offsets": [[offset, offset + 1]],
            }
            for offset in range(3)
        ],
    ]


def test_matched_control_provenance_requires_true_identity_and_nonzero_offsets() -> None:
    valid = _matched_arms()
    assert _matched_control_provenance_valid(valid) is True

    mutations = []
    missing_type = deepcopy(valid)
    missing_type[0]["identity_source_factor_types"] = []
    mutations.append(missing_type)
    wrong_type = deepcopy(valid)
    wrong_type[0]["identity_source_factor_types"] = ["lane"]
    mutations.append(wrong_type)
    wrong_region = deepcopy(valid)
    wrong_region[0]["identity_source_regions"] = ["left"]
    mutations.append(wrong_region)
    zero_offset = deepcopy(valid)
    zero_offset[1]["spatial_offsets"] = [[0, 0]]
    mutations.append(zero_offset)
    malformed_offset = deepcopy(valid)
    malformed_offset[1]["spatial_offsets"] = [[1]]
    mutations.append(malformed_offset)
    unavailable = deepcopy(valid)
    unavailable[0]["available_sample_count"] = 0
    mutations.append(unavailable)

    assert all(_matched_control_provenance_valid(arms) is False for arms in mutations)


def _json_payloads() -> dict[str, dict[str, object]]:
    payloads = {name: {"available": True, "epoch": 0} for name in ICDOR_EPOCH_JSON_FILES}
    payloads["metrics_summary.json"] = {
        "available": True,
        "epoch": 0,
        "sample_count": 2,
        "raw": {"Act_mF1": 0.5, "Exp_mF1": 0.4},
        "deploy_fixed": {"Act_mF1": 0.5, "Exp_mF1": 0.4},
        "test_oracle_diagnostic": {"Act_mF1": 0.6, "Exp_mF1": 0.5},
    }
    payloads["target_transfer_summary.json"] = {
        "available": True, "epoch": 0, "source_split": "train_audit",
        "schema_version": "mosaic_target_transfer.v2", "target_count": 2,
    }
    payloads["visual_audit_manifest.json"] = {
        "schema": "icdor_visual_audit_v1", "source_split": "train_audit",
        "sample_count": 1, "fixed_sample_ids": ["audit.jpg"],
        "matched_random_control": "same_factor_equal_mass_spatial_roll",
        "samples": [{"file_name": "audit.jpg", "factor_mask_files": ["masks/a.pt"],
                     "matched_random_factor_mask_files": ["masks/r.pt"]}],
    }
    return payloads


def _jsonl_payloads() -> dict[str, list[dict[str, object]]]:
    rows = {name: [{"available": True, "epoch": 0, "source_split": "test"}] for name in ICDOR_EPOCH_JSONL_FILES}
    rows["calibration_stats.jsonl"] = [{"available": True, "epoch": 0, "source_split": "train_calib"}]
    rows["reason_dual_observation_stats.jsonl"] = [
        {
            "available": True,
            "epoch": 0,
            "source_split": "train_audit",
            "audit": "hidden_recovery",
            "mode": mode,
            "hide_fraction": fraction,
            "evaluation_only": True,
        }
        for mode in ("mcar", "mar", "mnar")
        for fraction in (0.10, 0.30, 0.50)
    ]
    rows["gradient_ownership.jsonl"] = [
        {"epoch": 0, "step": 0, "loss": "loss_action_total", "owner_group": "reason_adapter",
         "grad_norm": 0.0, "finite": True, "cosine_with_action_base": None},
        {"epoch": 0, "step": 0, "loss": "loss_reason_total", "owner_group": "action_adapter",
         "grad_norm": 0.0, "finite": True, "cosine_with_action_base": None},
    ]
    rows["target_transfer_stats.jsonl"] = [{
        "available": True, "epoch": 0, "source_split": "train_audit",
        "factor_id": "f", "target_id": "a", "tet": 0.1, "tes": 0.05,
        "tes_identity": 0.03, "tes_spatial": 0.02,
        "cca": 0.01, "ap_delta": 0.02, "matched_control_arms": _matched_arms(),
    }]
    return rows


def _logits() -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in ICDOR_LOGIT_FILES:
        if name == "action_labels.pt":
            result[name] = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
        elif name == "reason_labels.pt":
            result[name] = torch.zeros(2, 21)
        elif "action" in name:
            result[name] = torch.randn(2, 4)
        elif "observation_model_prob" in name:
            result[name] = torch.sigmoid(torch.randn(2, 21))
        else:
            result[name] = torch.randn(2, 21)
    return result


def test_artifact_writer_requires_full_icdor_schema_and_validates_it(tmp_path: Path) -> None:
    initialize_icdor_run_artifacts(
        tmp_path,
        manifest={
            "git_head": "f" * 40,
            "pretrained_sha256": "a" * 64,
            "direct_image": True,
            "feature_cache": False,
            "token_compression": "none",
            "best_selection_split": "test",
        },
        config={"experiment": {"name": "acpr_mosaic_trust_v3_icdor"}},
        source_manifest={"base_head": "f" * 40},
        split_manifest={"split_sha256": "b" * 64},
        runtime_selection={"batch_size": 4, "grad_accum": 8},
        factor_certificate={"sha256": "c" * 64, "source_split": "train_audit"},
        edge_admission={"sha256": "d" * 64, "source_split": "train_audit"},
    )
    write_icdor_epoch_artifacts(
        tmp_path,
        epoch=0,
        json_payloads=_json_payloads(),
        jsonl_payloads=_jsonl_payloads(),
        logits=_logits(),
        file_names=["a.jpg", "b.jpg"],
    )
    masks = tmp_path / "epoch_000" / "masks"
    masks.mkdir()
    original = torch.arange(16.0).view(4, 4)
    torch.save(original, masks / "a.pt")
    torch.save(torch.roll(original, shifts=(1, 1), dims=(-2, -1)), masks / "r.pt")

    result = validate_icdor_artifact_schema(tmp_path, epochs=[0], strict_semantics=True)

    assert result["pass"], result

    transfer_path = tmp_path / "epoch_000" / "target_transfer_stats.jsonl"
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    transfer["matched_control_arms"][0]["identity_source_factor_names"] = []
    transfer_path.write_text(json.dumps(transfer) + "\n", encoding="utf-8")
    invalid = validate_icdor_artifact_schema(tmp_path, epochs=[0], strict_semantics=True)
    assert invalid["pass"] is False
    assert any("target transfer rows" in error for error in invalid["semantic_errors"])


def test_artifact_writer_rejects_missing_branch_or_test_leaked_threshold(tmp_path: Path) -> None:
    initialize_icdor_run_artifacts(
        tmp_path,
        manifest={
            "git_head": "f" * 40,
            "pretrained_sha256": "a" * 64,
            "direct_image": True,
            "feature_cache": False,
            "token_compression": "none",
            "best_selection_split": "test",
        },
        config={"experiment": {"name": "acpr_mosaic_trust_v3_icdor"}},
        source_manifest={"base_head": "f" * 40},
        split_manifest={"split_sha256": "b" * 64},
        runtime_selection={"batch_size": 4, "grad_accum": 8},
        factor_certificate={"sha256": "c" * 64, "source_split": "train_audit"},
        edge_admission={"sha256": "d" * 64, "source_split": "train_audit"},
    )
    rows = _jsonl_payloads()
    rows["calibration_stats.jsonl"] = [{"available": True, "epoch": 0, "source_split": "test"}]
    payloads = _json_payloads()
    payloads.pop("branch_metrics.json")

    try:
        write_icdor_epoch_artifacts(
            tmp_path,
            epoch=0,
            json_payloads=payloads,
            jsonl_payloads=rows,
            logits=_logits(),
            file_names=["a.jpg", "b.jpg"],
        )
    except ValueError as error:
        assert "schema" in str(error)
    else:
        raise AssertionError("writer accepted an incomplete IC-DOR branch artifact schema")


def test_strict_schema_rejects_fake_transfer_and_visual_controls(tmp_path: Path) -> None:
    initialize_icdor_run_artifacts(
        tmp_path,
        manifest={"git_head": "f" * 40, "pretrained_sha256": "a" * 64,
                  "direct_image": True, "feature_cache": False, "token_compression": "none",
                  "best_selection_split": "test"},
        config={"experiment": {"name": "acpr_mosaic_trust_v3_icdor"}},
        source_manifest={"base_head": "f" * 40}, split_manifest={"split_sha256": "b" * 64},
        runtime_selection={"batch_size": 4, "grad_accum": 8},
        factor_certificate={"sha256": "c" * 64, "source_split": "train_audit"},
        edge_admission={"sha256": "d" * 64, "source_split": "train_audit"},
    )
    payloads = _json_payloads()
    payloads["target_transfer_summary.json"] = {"available": False, "source_split": "train_audit"}
    payloads["visual_audit_manifest.json"]["matched_random_control"] = "next_factor"
    write_icdor_epoch_artifacts(tmp_path, epoch=0, json_payloads=payloads,
                                jsonl_payloads=_jsonl_payloads(), logits=_logits(),
                                file_names=["a.jpg", "b.jpg"])
    result = validate_icdor_artifact_schema(tmp_path, epochs=[0], strict_semantics=True)
    assert result["pass"] is False
    assert any("target transfer" in error for error in result["semantic_errors"])
    assert any("matched-random" in error for error in result["semantic_errors"])
