from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch

from fate_oia.utils.mosaic_icdor_artifacts import (
    ICDOR_EPOCH_JSON_FILES,
    ICDOR_EPOCH_JSONL_FILES,
    ICDOR_LOGIT_FILES,
    _gradient_firewall_rows_valid,
    initialize_icdor_run_artifacts,
    validate_icdor_artifact_schema,
    write_icdor_epoch_artifacts,
    write_icdor_adaptive_schedule_transition,
    _matched_control_provenance_valid,
)


def test_gradient_firewall_schema_requires_all_credo_owner_boundaries() -> None:
    rows = []
    for loss, owners in {
        "loss_action_total": {
            "factor_visual_pyramid", "factor_adapter", "factor_extractor", "factor_prototypes",
            "reason_visual_pyramid", "reason_adapter", "reason_visual_decoder",
            "reason_latent_decoder", "reason_observed_mixer", "observation_model",
        },
        "loss_reason_total": {
            "factor_visual_pyramid", "factor_adapter", "factor_extractor", "factor_prototypes",
            "action_visual_pyramid", "action_adapter", "action_visual_decoder", "action_router_rereader",
        },
    }.items():
        rows.extend({"loss": loss, "owner_group": owner, "grad_norm": 0.0, "finite": True} for owner in owners)

    assert _gradient_firewall_rows_valid(rows) is True
    rows[-1]["grad_norm"] = 1e-6
    assert _gradient_firewall_rows_valid(rows) is False


def _matched_arms() -> list[dict[str, object]]:
    common = {
        "available_sample_count": 2,
        "max_mass_error": 0.0,
        "max_overlap": 0.0,
        "control_support_method": "topk_continuous_evidence",
        "control_evidence_slots": 16,
        "selected_support_count_mean": 16.0,
        "selected_mass_fraction_mean": 0.25,
        "source_region_mass_total": 8.0,
        "selected_factor_indices": [0],
    }
    return [
        {
            **common,
            "control_type": "same_type_identity",
            "identity_source_factor_names": ["identity_factor"],
            "identity_source_factor_indices": [1],
            "identity_source_factor_types": ["object"],
            "identity_source_regions": ["front_center"],
            "identity_sources": [{"index": 1, "name": "identity_factor", "type": "object", "region": "front_center"}],
            "factor_type": "object",
            "region": "front_center",
            "spatial_offsets": [],
        },
        *[
            {
                **common,
                "control_type": "spatial_roll",
                "identity_source_factor_names": [],
                "identity_source_factor_indices": [],
            "identity_source_factor_types": [],
            "identity_source_regions": [],
            "identity_sources": [],
                "factor_type": "object",
                "region": "front_center",
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
    reused_selected_factor = deepcopy(valid)
    reused_selected_factor[0]["identity_source_factor_indices"] = [0]
    mutations.append(reused_selected_factor)
    unspecified_region = deepcopy(valid)
    unspecified_region[0]["region"] = "unspecified"
    mutations.append(unspecified_region)
    missing_sources = deepcopy(valid)
    missing_sources[0]["identity_sources"] = []
    mutations.append(missing_sources)
    wrong_source_type = deepcopy(valid)
    wrong_source_type[0]["identity_sources"][0]["type"] = "lane"
    mutations.append(wrong_source_type)
    reused_source = deepcopy(valid)
    reused_source[0]["identity_sources"][0]["index"] = 0
    mutations.append(reused_source)

    assert all(_matched_control_provenance_valid(arms) is False for arms in mutations)


def test_matched_control_provenance_allows_multiple_aligned_wrong_identity_sources() -> None:
    arms = _matched_arms()
    arms[0]["identity_source_factor_names"] = ["identity_factor", "another_identity_factor"]
    arms[0]["identity_source_factor_indices"] = [1, 2]
    arms[0]["identity_sources"] = [
        {"index": 1, "name": "identity_factor", "type": "object", "region": "front_center"},
        {"index": 2, "name": "another_identity_factor", "type": "object", "region": "front_center"},
    ]
    assert _matched_control_provenance_valid(arms) is True


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
    payloads["mechanism_summary.json"] = {
        "schema_version": "mosaic_icdor_mechanism_summary.v2",
        "epoch": 0,
        "available": True,
        "missing_evidence": [],
        "continuous_credibility": {"available": True, "content_beats_prior_factor_count": 1},
        "fine_transport": {
            "available": True, "fine_mask_delta_mean": 0.01,
            "fine_off_action_shadow_delta_abs_mean": 0.02,
            "fine_off_reason_latent_delta_abs_mean": 0.02,
        },
        "reason_transport": {
            "available": True, "route_off_logit_delta_abs_mean": 0.02,
            "shuffle_logit_delta_abs_mean": 0.01, "visual_exp_map": 0.4,
            "final_exp_map": 0.4,
            "no_lane_absence_polarity": {
                "available": True, "contract": "observability_times_absence",
            },
        },
        "action_shadow": {
            "available": True, "route_to_visual_rms_ratio_mean": 0.02,
            "final_act_map": 0.5, "final_visual_exact": True,
        },
        "pu": {"available": True, "schedule_enabled": True},
        "target_effectiveness": {"available": True},
        "gradient_firewall": {"available": True, "pass": True},
        "interpretation": {
            "learning_access": "continuous_credibility_and_shadow_routes",
            "deployment_admission": "edge_audit_only",
            "certificate_role": "final_reporting_only",
        },
    }
    payloads["factor_audit.json"] = {
        "source_split": "audit_visual",
        "factor_stats": {
            "road": {
                "counts": {}, "scores": {}, "prototype": {}, "bootstrap_lcb95": {},
            },
        },
    }
    payloads["target_transfer_summary.json"] = {
        "available": True, "epoch": 0, "source_split": "audit_target",
        "schema_version": "mosaic_target_transfer.v2", "pair_count": 1,
        "audit_level": "online",
    }
    payloads["visual_credibility.json"] = {
        "available": True, "epoch": 0, "source_split": "audit_visual", "credibility": [],
    }
    payloads["semantic_compatibility.json"] = {
        "available": True, "epoch": 0, "source_split": "audit_target",
        "audit_level": "online", "semantic_compatibility": [[0.5]],
    }
    payloads["target_utility.json"] = {
        "available": True, "epoch": 0, "source_split": "audit_target",
        "audit_level": "online", "action_target_utility": [[0.1]],
    }
    payloads["visual_audit_manifest.json"] = {
        "schema": "icdor_visual_audit_v1", "source_split": "audit_visual",
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
            "available": True, "epoch": 0, "split": "test", "reason_id": reason_id,
            "residual_alpha_mean": 0.1, "escape_weight_mean": 0.2,
            "allowed_factor_mass_mean": 0.3, "disallowed_factor_mass_mean": 0.0,
            "reason_factor_mask_area_mean": 0.2, "reason_factor_mask_entropy": 0.4,
            "semantic_compatibility_mean": 0.5, "absence_factor_mass_mean": 0.1,
            "absence_negative_evidence_mean": 0.1,
        }
        for reason_id in range(21)
    ] + [
        {
            "available": True,
            "epoch": 0,
            "source_split": "audit_target",
            "audit": "hidden_recovery",
            "mode": mode,
            "hide_fraction": fraction,
            "evaluation_only": True,
        }
        for mode in ("mcar", "mar", "mnar")
        for fraction in (0.10, 0.30, 0.50)
    ]
    rows["gradient_ownership.jsonl"] = [
        {"epoch": 0, "step": 0, "loss": loss, "owner_group": owner,
         "grad_norm": 0.0, "finite": True}
        for loss, owners in {
            "loss_action_total": {
                "factor_visual_pyramid", "factor_adapter", "factor_extractor", "factor_prototypes",
                "reason_visual_pyramid", "reason_adapter", "reason_visual_decoder",
                "reason_latent_decoder", "reason_observed_mixer", "observation_model",
            },
            "loss_reason_total": {
                "factor_visual_pyramid", "factor_adapter", "factor_extractor", "factor_prototypes",
                "action_visual_pyramid", "action_adapter", "action_visual_decoder", "action_router_rereader",
            },
        }.items()
        for owner in owners
    ]
    rows["target_transfer_stats.jsonl"] = [{
        "available": True, "epoch": 0, "source_split": "audit_target", "audit_level": "online",
        "factor_id": "f", "target_id": "a", "tet": 0.1, "tes": 0.05,
        "tes_identity": 0.03, "tes_spatial": 0.02,
        "cca": 0.01, "ap_delta": 0.02, "matched_control_arms": _matched_arms(),
    }]
    rows["credibility_stats.jsonl"] = [{
        "epoch": 0, "split": "test", "factor_id": 0,
        "cV_mean": 0.2, "cV_p50": 0.2, "cV_p95": 0.3,
        "cV_ema_mean": 0.2, "cV_nonzero_rate": 1.0, "cV_route_effective_mean": 0.2,
    }]
    rows["fine_transport_stats.jsonl"] = [{
        "epoch": 0, "split": "test", "typed_coordinates_present": True,
        "fine_mask_delta_mean": 0.01, "fine_mask_delta_max": 0.02,
        "anchor_separation_mean": 0.3,
        "fine_off_action_shadow_delta_abs_mean": 0.02,
        "fine_off_reason_latent_delta_abs_mean": 0.02,
        "coarse_off_action_shadow_delta_abs_mean": 0.01,
        "coarse_off_reason_latent_delta_abs_mean": 0.01,
    }]
    rows["route_ownership.jsonl"] = [{
        "epoch": 0, "split": "test", "summary": "per_action_route_effect",
        "route_mode": "shadow", "action_final_visual_equal": True,
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
            "credo_version": "v5_credo_map",
            "pilot": False,
        },
        config={"experiment": {"name": "acpr_mosaic_trust_v5_credo_map"}},
        source_manifest={"base_head": "f" * 40},
        split_manifest={
            "split_sha256": "b" * 64,
            "file_names": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "audit_visual_indices": [0], "audit_target_indices": [1],
            "train_core_indices": [2], "train_calib_indices": [3],
        },
        runtime_selection={"batch_size": 4, "grad_accum": 8},
        factor_certificate={"sha256": "c" * 64, "source_split": "audit_visual"},
        edge_admission={"sha256": "d" * 64, "source_split": "audit_target"},
    )
    write_icdor_adaptive_schedule_transition(tmp_path, {
        "epoch": 0, "state_before": "JOINT_SHADOW", "state_after": "JOINT_SHADOW",
        "state_epochs_before": 0, "state_epochs_after": 1, "ready": True,
        "failed_closed": False,
        "readiness": {
            "train_core": {"source_split": "train_core", "finite": True},
            "train_audit": {"source_split": "train_audit", "finite": True},
            "train_calib": {"source_split": "train_calib", "finite": True},
        },
        "certificate_sha256": "c" * 64, "edge_admission_sha256": "d" * 64,
    })
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
    assert _matched_control_provenance_valid(transfer["matched_control_arms"]) is False


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
