from __future__ import annotations

from pathlib import Path
import hashlib

import pytest
import torch
from torch import nn

from fate_oia.engine.train_acpr_mosaic_trust_icdor import (
    _parameter_group_gradient_audit,
    _certificate_readiness_metrics,
    _edge_statistics_from_audit,
    _edge_and_branch_readiness_metrics,
    _factor_audit_integrity_metrics,
    _factor_audit_rows,
    _build_rank_queues,
    _load_resume,
    _pending_evidence_document,
    _pilot_semantic_validation,
    _restore_phase_trainability,
    _save_checkpoint,
    _target_transfer_directions,
    _validate_review_evidence_bindings,
    _write_provisional_certificate_this_epoch,
    assert_icdor_gradient_firewall,
    apply_icdor_consolidation,
    apply_icdor_factor_branch_freeze,
    build_icdor_parameter_ownership,
    build_icdor_optimizer,
    build_icdor_model,
    compute_icdor_training_losses,
    load_config,
)


def test_resume_happens_before_fresh_artifact_initialization() -> None:
    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    main_index = source.index("def main")
    resume_index = source.index("_load_resume(", main_index)
    initialize_index = source.index("initialize_icdor_run_artifacts(", main_index)
    assert resume_index < initialize_index


def test_full_cli_cannot_bypass_review_pass() -> None:
    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert "if not args.pilot and not args.require_review_pass" in source
    assert "full training requires --require_review_pass" in source


def test_full_cli_rejects_review_pass_bound_to_a_different_split() -> None:
    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert 'review_payload.get("split_sha256") != split_stats["split_sha256"]' in source
    assert "REVIEW_PASS split hash mismatch" in source


def test_review_pass_rehashes_ignored_pilot_evidence_before_launch(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / file_name
        for name, file_name in {
            "pilot_gate": "pilot_gate.json",
            "factor_certificate": "factor_certificate.json",
            "edge_admission": "edge_admission.json",
            "final_remediation_plan": "final_remediation.md",
            "audit_addendum": "audit_addendum.md",
        }.items()
    }
    for name, path in paths.items():
        path.write_text(f'{{"artifact":"{name}"}}\n', encoding="utf-8")
    payload = {
        "evidence_files": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
            }
            for name, path in paths.items()
        }
    }
    _validate_review_evidence_bindings(payload)

    for name, path in paths.items():
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        with pytest.raises(RuntimeError, match=name):
            _validate_review_evidence_bindings(payload)
        path.write_bytes(original)


def test_split_hash_is_checked_only_after_split_manifest_is_built() -> None:
    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    main_index = source.index("def main")
    loaders_index = source.index("build_icdor_loaders(", main_index)
    split_check_index = source.index('review_payload.get("split_sha256")', main_index)
    assert loaders_index < split_check_index


def test_credo_trainer_uses_disjoint_visual_and_target_audit_loaders() -> None:
    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")

    assert "audit_visual_loader" in source
    assert "audit_target_loader" in source
    assert 'source_split="audit_visual"' in source
    assert "refresh_model_visual_credibility" in source


def test_checkpoint_persists_all_rng_state_for_resume() -> None:
    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    for token in ("python_rng_state", "torch_rng_state", "cuda_rng_state_all"):
        assert token in source
from fate_oia.engine.mosaic_icdor_schedule import ICDORPhase
from fate_oia.optim.mosaic_action_pareto_admission import MOSAICActionParetoAdmission
from fate_oia.optim.mosaic_soft_rank_queue import MOSAICSoftRankQueue


class _ProtocolModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.factor_visual_pyramid = nn.Linear(2, 2)
        self.factor_adapter = nn.Linear(2, 2)
        self.factor_extractor = nn.Module()
        self.factor_extractor.prototype_bank = nn.Linear(2, 2)
        self.factor_extractor.measurement = nn.Linear(2, 2)
        self.action_visual_pyramid = nn.Linear(2, 2)
        self.action_adapter = nn.Linear(2, 2)
        self.action_visual_decoder = nn.Linear(2, 2)
        self.action_router = nn.Linear(2, 2)
        self.action_rereader = nn.Linear(2, 2)
        self.reason_visual_pyramid = nn.Linear(2, 2)
        self.reason_adapter = nn.Linear(2, 2)
        self.reason_visual_decoder = nn.Linear(2, 2)
        self.reason_latent_decoder = nn.Linear(2, 2)
        self.reason_observed_mixer = nn.Linear(2, 2)
        self.observation_model = nn.Linear(2, 2)
        self.threshold_head = nn.Linear(2, 2)


def test_formal_config_is_test_only_and_disallows_legacy_state_path() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml"
    config = load_config(config_path)

    assert config["training"]["epochs"] == 12
    assert config["evaluation"]["eval_splits"] == ["test"]
    assert config["backbone"]["feature_cache"] is False
    assert config["backbone"]["token_compression"] == "none"
    assert config["model"]["formal_class"] == "MOSAICTrustICDORModel"
    assert config["model"]["action_set_final"] is False


def test_parameter_ownership_is_unique_and_has_three_independent_lanes() -> None:
    ownership, groups = build_icdor_parameter_ownership(_ProtocolModel())

    assert set(groups) == {
        "factor_visual_pyramid",
        "action_visual_pyramid",
        "reason_visual_pyramid",
        "factor_adapter",
        "factor_extractor",
        "factor_prototypes",
        "action_adapter",
        "action_visual_decoder",
        "action_router_rereader",
        "reason_adapter",
        "reason_visual_decoder",
        "reason_latent_decoder",
        "reason_observed_mixer",
        "observation_model",
        "threshold_head",
    }
    names = [entry["full_name"] for entry in ownership]
    assert len(names) == len(set(names))
    assert any(name.startswith("factor_visual_pyramid") for name in names)
    assert any(name.startswith("action_visual_pyramid") for name in names)
    assert any(name.startswith("reason_visual_pyramid") for name in names)
    assert "visual_pyramid" not in groups
    assert all(entry["allowed_losses"] for entry in ownership)
    assert {entry["owner_group"] for entry in ownership} == set(groups)
    factor_entries = [entry for entry in ownership if entry["owner_group"] == "factor_visual_pyramid"]
    action_entries = [entry for entry in ownership if entry["owner_group"] == "action_visual_pyramid"]
    reason_entries = [entry for entry in ownership if entry["owner_group"] == "reason_visual_pyramid"]
    assert factor_entries and action_entries and reason_entries
    assert all(entry["allowed_losses"] == ["factor"] for entry in factor_entries)
    assert all(entry["allowed_losses"] == ["action_base"] for entry in action_entries)
    assert all(entry["allowed_losses"] == ["reason_observed"] for entry in reason_entries)
    reason_adapter_entries = [entry for entry in ownership if entry["owner_group"] == "reason_adapter"]
    assert reason_adapter_entries
    assert all(entry["allowed_losses"] == ["reason_observed"] for entry in reason_adapter_entries)


def test_ownership_rejects_unassigned_or_ambiguous_trainable_parameter() -> None:
    model = _ProtocolModel()
    model.unowned = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="unassigned"):
        build_icdor_parameter_ownership(model)


def test_certificate_freeze_disables_only_the_factor_branch() -> None:
    model = _ProtocolModel()
    apply_icdor_factor_branch_freeze(model)

    assert all(not parameter.requires_grad for parameter in model.factor_visual_pyramid.parameters())
    assert all(not parameter.requires_grad for parameter in model.factor_adapter.parameters())
    assert all(not parameter.requires_grad for parameter in model.factor_extractor.parameters())
    assert all(parameter.requires_grad for parameter in model.action_adapter.parameters())
    assert all(parameter.requires_grad for parameter in model.reason_adapter.parameters())


def test_formal_loss_surface_calls_factor_action_reason_and_preserves_firewalls() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml")
    model = build_icdor_model(config, use_mock_dino=True, mock_dim=32)
    model.train()
    images = torch.randn(2, 3, 360, 640)
    output = model(images, route_mode="off", latent_enabled=False, return_masks=True)
    second = {key: output[key] for key in ("factor_presence_prob", "factor_visibility_prob", "factor_soft_masks")}
    factor_count = len(model.ontology["factors"])
    observations = {
        "presence_target": torch.zeros(2, factor_count),
        "visibility_target": torch.zeros(2, factor_count),
        "presence_known_mask": torch.zeros(2, factor_count),
        "visibility_known_mask": torch.zeros(2, factor_count),
        "weak_negative_mask": torch.zeros(2, factor_count),
        "geometry_masks": torch.zeros(2, factor_count, 45, 80),
        "geometry_known_mask": torch.zeros(2, factor_count),
    }
    batch = {
        "action": torch.randint(0, 2, (2, 4)).float(),
        "reason": torch.randint(0, 2, (2, 21)).float(),
        "file_name": ["a.jpg", "b.jpg"],
    }
    phase = ICDORPhase("visual_foundation", "off", False, True, False, False, False)
    losses = compute_icdor_training_losses(
        model, output, second, batch, observations, phase,
        MOSAICActionParetoAdmission(), MOSAICSoftRankQueue(4, capacity=8), MOSAICSoftRankQueue(21, capacity=8),
        hidden_mask=torch.zeros(2, 21, dtype=torch.bool),
    )
    assert torch.isfinite(losses["loss_total"])
    ownership, _ = build_icdor_parameter_ownership(model)
    gradient_rows = _parameter_group_gradient_audit(
        losses,
        ownership,
        model,
        epoch=0,
        step=0,
    )
    assert_icdor_gradient_firewall(gradient_rows)
    losses["loss_total"].backward()
    assert any(parameter.grad is not None for parameter in model.action_adapter.parameters())
    assert any(parameter.grad is not None for parameter in model.factor_adapter.parameters())


def test_pending_evidence_is_explicit_and_never_claims_train_audit_completion() -> None:
    factor = _pending_evidence_document("factor_certificate", build_epoch=4)
    edge = _pending_evidence_document("edge_admission", build_epoch=6)

    assert factor == {
        "artifact": "factor_certificate",
        "status": "pending",
        "available": False,
        "source_split": None,
        "build_epoch": 4,
        "reason": "scheduled_train_audit_collection_not_completed",
    }
    assert edge["source_split"] is None
    assert edge["available"] is False


def test_edge_audit_adapter_preserves_real_lcb_and_ap_values() -> None:
    payload = {
        "source_split": "train_audit",
        "edge_stats": {
            "support:f0->stop": {
                "factor": "f0",
                "action": "stop",
                "direction": "support",
                "available": True,
                "metrics": {"cca": 0.75, "isolated_edge_ap": 0.81, "visual_ap": 0.80},
                "bootstrap_lcb95": {
                    "signed_effect": 0.02, "tet": 0.03, "tes": 0.01,
                    "tes_identity": 0.008, "tes_spatial": 0.012,
                },
                "matched_counts": {"factor_on": 80, "factor_off": 80, "equal_mass_random": 80},
            }
        },
    }
    converted = _edge_statistics_from_audit(payload)
    record = converted[("support", "f0", "stop")]
    assert record.valid_samples == 80
    assert record.signed_effect_lcb95 == pytest.approx(0.02)
    assert record.isolated_edge_ap == pytest.approx(0.81)
    assert record.visual_ap == pytest.approx(0.80)


def test_factor_rows_preserve_audit_visual_counts_scores_and_bootstrap() -> None:
    payload = {
        "source_split": "audit_visual",
        "factor_stats": {
            "traffic_light_visible": {
                "counts": {"confirmed_positive": 12, "reliable_negative": 20, "weak_negative": 3, "unknown": 5},
                "scores": {"full": 0.7, "content_only": 0.6, "prior_only": 0.2},
                "prototype": {"effective_count": 2.1, "dominant_rate": 0.2, "dead_count": 0},
                "bootstrap_lcb95": {"full_minus_prior_only": 0.03},
            }
        },
    }
    rows = _factor_audit_rows(payload, epoch=3)
    assert rows[0]["source_split"] == "audit_visual"
    assert rows[0]["counts"]["confirmed_positive"] == 12
    assert rows[0]["prototype"]["effective_count"] == pytest.approx(2.1)


def test_phase_trainability_restores_then_applies_factor_freeze() -> None:
    model = _ProtocolModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase = ICDORPhase("formal", "shadow", True, False, False, False, True)

    _restore_phase_trainability(model, phase)

    assert all(not parameter.requires_grad for parameter in model.factor_visual_pyramid.parameters())
    assert all(not parameter.requires_grad for parameter in model.factor_adapter.parameters())
    assert all(parameter.requires_grad for parameter in model.action_adapter.parameters())
    assert all(parameter.requires_grad for parameter in model.threshold_head.parameters())


def test_formal_main_is_not_a_disabled_placeholder() -> None:
    source = (Path(__file__).parents[1] / "fate_oia" / "engine" / "train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert "execution is intentionally disabled" not in source
    for token in (
        "collect_factor_audit(",
        "build_and_write_factor_certificate(",
        "collect_edge_intervention_audit(",
        "build_edge_admission(",
        "fit_icdor_calibration(",
        "evaluate_icdor(",
        "write_icdor_epoch_artifacts(",
        "collect_joint_target_transfer_metrics(",
        "export_visual_audit(",
        "collect_action_pareto_audit(",
        "checkpoint_latest.pth",
    ):
        assert token in source


def test_foreground_launcher_has_non_circular_gate_order_and_exact_protocol() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "FATE_OIA_acpr_mosaic_trust_v3_icdor_foreground.ps1").read_text(encoding="utf-8")
    profile_block = source.split('"profile" {', 1)[1].split('"pilot" {', 1)[0]
    pilot_block = source.split('"pilot" {', 1)[1].split('"full" {', 1)[0]
    full_block = source.split('"full" {', 1)[1]
    assert "Assert-ReviewPass" not in profile_block
    assert "--config" in profile_block and "--device" in profile_block
    assert "--pilot" in pilot_block and "--epochs" in pilot_block and '"6"' in pilot_block
    expected_pilot_limits = {
        "--max_train_samples": '"2048"',
        "--max_audit_samples": '"512"',
        "--max_calib_samples": '"512"',
        "--max_test_samples": '"512"',
    }
    for option, value in expected_pilot_limits.items():
        assert option in pilot_block and value in pilot_block
    assert "--write_review_pass" in pilot_block and "pilot_gate.json" in source
    assert "--final_remediation_plan" in pilot_block and "$FinalRemediationPlan" in pilot_block
    assert "--audit_addendum" in pilot_block and "$AuditAddendum" in pilot_block
    assert "Assert-ReviewPass" in full_block
    assert "--require_review_pass" in full_block
    assert "--runtime_selection" in full_block
    for evidence_name in ("pilot_gate", "factor_certificate", "edge_admission", "final_remediation_plan", "audit_addendum"):
        assert evidence_name in full_block or evidence_name in source[source.index("function Assert-ReviewPass"):source.index("function Invoke-ForegroundPython")]
    assert "evidence hash mismatch" in source


def test_rank_queues_are_constructed_on_the_requested_device() -> None:
    action, reason = _build_rank_queues(capacity=17, device=torch.device("cpu"))
    assert action.capacity == 17 and reason.capacity == 17
    assert action.logit_buffer.device.type == "cpu"
    assert reason.logit_buffer.device.type == "cpu"


def test_target_transfer_directions_do_not_invent_unauthorized_edges() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml")
    model = build_icdor_model(config, use_mock_dino=True, mock_dim=32)
    action, reason = _target_transfer_directions(model.ontology)
    assert {value for row in action for value in row} <= {"support", "veto", "none"}
    assert {value for row in reason for value in row} <= {"support", "veto", "none"}
    assert any(value == "none" for row in action for value in row)
    assert any(value == "none" for row in reason for value in row)


def test_foundation_rebuilds_provisional_certificate_until_policy_freezes_it() -> None:
    assert _write_provisional_certificate_this_epoch(
        write_provisional=True, freeze_certificate=False
    ) is True
    assert _write_provisional_certificate_this_epoch(
        write_provisional=False, freeze_certificate=True
    ) is False
    assert _write_provisional_certificate_this_epoch(
        write_provisional=False, freeze_certificate=False
    ) is False


def test_certificate_readiness_is_derived_from_real_tiers_and_routes() -> None:
    ontology = {
        "action_names": ["forward", "stop", "left", "right"],
        "action_routes": {
            "forward": {"support": [{"factor": "f0"}], "veto": []},
            "stop": {"support": [{"factor": "f1"}], "veto": []},
            "left": {"support": [{"factor": "f2"}], "veto": []},
            "right": {"support": [{"factor": "f3"}], "veto": []},
        },
        "reason_routes": {
            0: {"direct_factors": ["f0"], "latent_factors": [], "contradiction_factors": []},
            1: {"direct_factors": [], "latent_factors": ["f4"], "contradiction_factors": []},
            2: {"direct_factors": ["f5"], "latent_factors": [], "contradiction_factors": []},
        },
    }
    certificate = {
        "entries": {
            "f0": {"tier": "certified"},
            "f1": {"tier": "certified"},
            "f2": {"tier": "abstained"},
            "f3": {"tier": "certified"},
            "f4": {"tier": "reason_only"},
            "f5": {"tier": "abstained"},
        }
    }
    previous = {name: row["tier"] for name, row in certificate["entries"].items()}
    factor_stats = {
        name: {
            "scores": {"presence_variance": 0.1, "visibility_variance": 0.1},
            "prototype": {"dominant_rate": 0.2},
        }
        for name in certificate["entries"]
    }

    metrics = _certificate_readiness_metrics(
        certificate=certificate,
        previous_tiers=previous,
        factor_stats=factor_stats,
        ontology=ontology,
    )

    assert metrics["certified_route_group_count_per_action"] == [1, 1, 0, 1]
    assert metrics["reachable_reason_count"] == 2
    assert metrics["certificate_tier_jaccard"] == pytest.approx(1.0)
    assert metrics["saturation_fraction"] == pytest.approx(0.0)


def test_certificate_readiness_does_not_treat_a_sha_as_real_eligibility() -> None:
    ontology = {
        "action_names": ["forward", "stop", "left", "right"],
        "action_routes": {
            name: {"support": [{"factor": "f0"}], "veto": []}
            for name in ("forward", "stop", "left", "right")
        },
        "reason_routes": {0: {"direct_factors": ["f0"], "latent_factors": [], "contradiction_factors": []}},
    }
    metrics = _certificate_readiness_metrics(
        certificate={"sha256": "PRESENT", "entries": {"f0": {"tier": "abstained"}}},
        previous_tiers=None,
        factor_stats={
            "f0": {
                "scores": {"presence_variance": 0.0, "visibility_variance": 0.0},
                "prototype": {"dominant_rate": 1.0},
            }
        },
        ontology=ontology,
    )
    assert metrics["certified_route_group_count_per_action"] == [0, 0, 0, 0]
    assert metrics["reachable_reason_count"] == 0
    assert metrics["certificate_tier_jaccard"] == 0.0
    assert metrics["saturation_fraction"] == 1.0


def test_shadow_readiness_uses_accepted_edge_contents_not_an_artifact_sha() -> None:
    branch = {
        "action_shadow_ap_delta": [0.0, 0.0, 0.0, 0.0],
        "action_route_ap_delta": [0.0, 0.0, 0.0, 0.0],
        "exp_map_delta_vs_visual": 0.01,
        "factor_shuffle_degrades_reason": True,
        "hidden_recovery_margin": 0.02,
        "route_strength_ratio": 0.05,
        "disallowed_route_invariance": True,
        "train_audit_joint": 0.5,
        "exp_map_delta_vs_entry": 0.01,
    }
    empty = _edge_and_branch_readiness_metrics(
        edge_document={"sha256": "EXISTS", "entries": {}},
        branch_metrics=branch,
        previous_best_train_audit_joint=0.4,
    )
    assert empty["true_edge_count_per_action"] == [0, 0, 0, 0]
    assert empty["tet_lcb95"] == 0.0
    assert empty["tes_lcb95"] == 0.0
    assert empty["cca"] == 0.0

    entries = {}
    for action in ("forward", "stop", "left", "right"):
        entries[f"support:f0:{action}"] = {
            "target": action,
            "accepted": True,
            "metrics": {
                "tet_lcb95": 0.02,
                "tes_lcb95": 0.01,
                "tes_identity_lcb95": 0.008,
                "tes_spatial_lcb95": 0.009,
                "cca": 0.7,
            },
        }
    real = _edge_and_branch_readiness_metrics(
        edge_document={"entries": entries},
        branch_metrics=branch,
        previous_best_train_audit_joint=0.4,
    )
    assert real["true_edge_count_per_action"] == [1, 1, 1, 1]
    assert real["tet_lcb95"] == pytest.approx(0.02)
    assert real["tes_lcb95"] == pytest.approx(0.01)
    assert real["cca"] == pytest.approx(0.7)
    assert real["train_audit_improved"] is True


def test_factor_audit_readiness_requires_collector_integrity_proof() -> None:
    valid = _factor_audit_integrity_metrics({
        "source_split": "train_audit",
        "factor_stats": {"f0": {"counts": {"unknown": 3}}},
        "audit_integrity": {
            "collector_completed": True,
            "exception": None,
            "unknown_policy": "excluded_from_binary_metrics",
            "unknown_rows_total": 3,
            "unknown_rows_in_metric_total": 0,
        },
    })
    assert valid == {
        "factor_audit_complete": True,
        "factor_audit_exception": False,
        "unknown_abstained": True,
    }

    missing_proof = _factor_audit_integrity_metrics({
        "source_split": "train_audit",
        "factor_stats": {"f0": {"counts": {"unknown": 3}}},
    })
    assert missing_proof["factor_audit_complete"] is False
    assert missing_proof["factor_audit_exception"] is True
    assert missing_proof["unknown_abstained"] is False


def test_pilot_gate_requires_real_state_transitions_and_accepted_edges() -> None:
    split_readiness = {
        "train_core": {"source_split": "train_core", "finite": True},
        "train_calib": {"source_split": "train_calib", "finite": True},
    }
    foundation = {
        "factor_audit_complete": True,
        "factor_audit_exception": False,
        "unknown_abstained": True,
        "certified_route_group_count_per_action": [1, 1, 1, 1],
        "reachable_reason_count": 15,
        "certificate_tier_jaccard": 0.95,
        "saturation_fraction": 0.1,
        "diagnostics_finite": True,
    }
    shadow = {
        "source_split": "train_audit",
        "continuous_credibility_available": True,
        "observable_cV_gt_030": 6,
        "exp_map_delta_vs_visual": 0.0,
        "factor_shuffle_degrades_reason": True,
        "hidden_recovery_margin": 0.02,
        "route_strength_ratio": 0.05,
        "action_shadow_ap_delta": [0.0] * 4,
        "true_edge_count_per_action": [1, 1, 0, 0],
        "disallowed_route_invariance": True,
        "diagnostics_finite": True,
    }
    history = [
        {"state_before": "FOUNDATION", "state_after": "FOUNDATION", "ready": True, "readiness": {**split_readiness, "train_audit": foundation}},
        {"state_before": "FOUNDATION", "state_after": "DUAL_REASON_SHADOW", "ready": True, "readiness": {**split_readiness, "train_audit": foundation}},
        {"state_before": "DUAL_REASON_SHADOW", "state_after": "DUAL_REASON_SHADOW", "ready": True, "readiness": {**split_readiness, "train_audit": shadow}},
        {"state_before": "DUAL_REASON_SHADOW", "state_after": "SAFE_JOINT", "ready": True, "readiness": {**split_readiness, "train_audit": shadow}},
        {"state_before": "SAFE_JOINT", "state_after": "SAFE_JOINT", "ready": True, "readiness": {"train_audit": {}}},
    ]
    schedule = {"history": history, "safe_joint_epochs": 1, "failed_closed": False}
    certificate = {"entries": {"f0": {"tier": "certified"}}}
    edge = {"entries": {
        "e0": {"accepted": True, "target": "forward"},
        "e1": {"accepted": True, "target": "stop"},
    }}
    result = _pilot_semantic_validation(schedule, certificate, edge)
    assert result["pass"] is True
    assert result["errors"] == []

    bad_split_history = [dict(row) for row in history]
    bad_split_history[1] = {
        **bad_split_history[1],
        "readiness": {
            **bad_split_history[1]["readiness"],
            "train_calib": {"source_split": "test", "finite": True},
        },
    }
    bad_split = _pilot_semantic_validation(
        {**schedule, "history": bad_split_history}, certificate, edge
    )
    assert bad_split["pass"] is False
    assert "foundation_transition_lacks_split_integrity" in bad_split["errors"]

    invalid = _pilot_semantic_validation(
        {"history": [], "safe_joint_epochs": 1, "failed_closed": False},
        {"sha256": "ONLY", "entries": {"f0": {"tier": "abstained"}}},
        {"sha256": "ONLY", "entries": {}},
    )
    assert invalid["pass"] is False
    assert "missing_foundation_to_shadow_transition" in invalid["errors"]
    assert "missing_or_invalid_shadow_learning_access" in invalid["errors"]


def test_credo_pilot_validates_learning_access_without_final_admission() -> None:
    """The pilot must not demand a certificate before it can learn evidence."""
    split_readiness = {
        "train_core": {"source_split": "train_core", "finite": True},
        "train_calib": {"source_split": "train_calib", "finite": True},
    }
    learning = {
        "source_split": "train_audit",
        "continuous_credibility_available": True,
        "observable_cV_gt_030": 6,
        "diagnostics_finite": True,
        "route_strength_ratio": 0.03,
        "factor_shuffle_degrades_reason": True,
        "hidden_recovery_margin": 0.02,
        "action_shadow_ap_delta": [0.0, 0.0, -0.001, 0.0],
        "disallowed_route_invariance": True,
    }
    schedule = {
        "history": [{
            "state_before": "FOUNDATION",
            "state_after": "DUAL_REASON_SHADOW",
            "ready": True,
            "readiness": {**split_readiness, "train_audit": learning},
        }],
        "safe_joint_epochs": 0,
        "failed_closed": False,
    }
    result = _pilot_semantic_validation(
        schedule,
        {"entries": {"f0": {"tier": "abstained"}}},
        {"entries": {}},
    )

    assert result["pass"] is True
    assert result["deployment_admission_ready"] is False


def test_checkpoint_roundtrip_restores_queues_and_pareto(tmp_path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    action, reason = _build_rank_queues(capacity=8, device=torch.device("cpu"))
    pareto = MOSAICActionParetoAdmission()
    action.enqueue(torch.ones(1, 4), torch.ones(1, 4), ["sample"])
    reason.enqueue(torch.ones(1, 21), torch.ones(1, 21), ["sample"])
    pareto.dual_variables.fill_(0.25)
    path = tmp_path / "checkpoint.pth"
    _save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=2,
        best_joint=0.5, certificate_sha256=None, edge_admission_sha256=None,
        config_sha256="CONFIG", split_sha256="SPLIT", action_queue=action,
        reason_queue=reason, pareto=pareto,
    )
    action._count.zero_(); action._count_value = 0
    reason._count.zero_(); reason._count_value = 0
    pareto.dual_variables.zero_()
    result = _load_resume(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        certificate_path=tmp_path / "certificate.json", edge_path=tmp_path / "edge.json",
        config_sha256="CONFIG", split_sha256="SPLIT", action_queue=action,
        reason_queue=reason, pareto=pareto,
    )
    assert result[:2] == (3, 0.5)
    assert action.count == 1 and reason.count == 1
    assert pareto.dual_variables.tolist() == pytest.approx([0.25] * 4)


def test_consolidation_freezes_propensity_and_reduces_only_decoder_router_lrs() -> None:
    model = _ProtocolModel()
    optimizer, _ = build_icdor_optimizer(
        model, load_config(Path(__file__).parents[1] / "configs" / "fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml")
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = {group["name"]: group["lr"] for group in optimizer.param_groups}
    apply_icdor_consolidation(model, optimizer, scheduler)
    after = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert all(not parameter.requires_grad for parameter in model.observation_model.parameters())
    assert after["action_visual_decoder"] == pytest.approx(before["action_visual_decoder"] * 0.2)
    assert after["action_adapter"] == before["action_adapter"]
