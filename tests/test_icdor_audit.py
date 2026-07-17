from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from fate_oia.engine.audit_acpr_mosaic_trust_icdor import (
    ICDORAuditError,
    build_review_pass,
    _existing_review_validation_requested,
    validate_real_factor_audit,
    protocol_hard_gate,
    verify_dynamic_forward_and_gradients,
)


def test_write_review_runtime_does_not_require_an_existing_review_pass() -> None:
    assert _existing_review_validation_requested(None, "runtime.json") is False
    assert _existing_review_validation_requested("review.json", "runtime.json") is True
    with pytest.raises(ICDORAuditError, match="runtime_selection"):
        _existing_review_validation_requested("review.json", None)


def test_real_factor_audit_accepts_honest_abstention_and_rejects_fake_zero() -> None:
    payload = {
        "source_split": "train_audit", "row_count": 512, "factor_count": 2,
        "factor_stats": {
            "available": {"evaluation_mode": "binary_confirmed", "metric_available": True, "presence_auprc": 0.7, "certificate_ceiling": "Certified"},
            "unknown": {"evaluation_mode": "unavailable", "metric_available": False, "presence_auprc": None, "certificate_ceiling": "Abstained"},
        },
    }
    assert validate_real_factor_audit(
        payload, expected_rows=512, expected_factors=2, expected_source="train_audit"
    )["pass"] is True
    payload["factor_stats"]["unknown"]["presence_auprc"] = 0.0
    with pytest.raises(ICDORAuditError, match="forged"):
        validate_real_factor_audit(
            payload, expected_rows=512, expected_factors=2, expected_source="train_audit"
        )


def _passing_remediation_gates(*, git_head: str = "abc"):
    names = (
        "CANONICAL_MULTIVIEW", "REAL_FACTOR_AUDIT", "HARD_MASK_INVARIANCE",
        "PARETO_FIREWALL", "HIDDEN_RECOVERY_NO_LEAKAGE", "MATCHED_CONTROL_CCA",
        "CONFIG_COVERAGE", "QUEUE_TIMING", "ADAPTIVE_SCHEDULE", "RUNTIME_PROFILE",
        "PILOT", "STRICT_ARTIFACT_VALIDATION",
    )
    return {name: {"pass": True, "gate": name, "git_head": git_head} for name in names}


def _bound_evidence_files() -> dict[str, dict[str, str]]:
    return {
        "pilot_gate": {"path": "pilot_gate.json", "sha256": "PILOT"},
        "factor_certificate": {"path": "factor_certificate.json", "sha256": "CERTIFICATE"},
        "edge_admission": {"path": "edge_admission.json", "sha256": "EDGE"},
        "final_remediation_plan": {"path": "final_remediation.md", "sha256": "PLAN"},
        "audit_addendum": {"path": "audit_addendum.md", "sha256": "ADDENDUM"},
    }


def test_certificate_audit_checks_builder_and_tier_logic_separately() -> None:
    source = Path("fate_oia/engine/audit_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert '"builder": _require_source_tokens' in source
    assert '"certificate_logic": _require_source_tokens' in source
    assert '"models" / "mosaic_factor_certificate.py"' in source


def test_functional_audit_cannot_blanket_pass_without_per_check_evidence() -> None:
    source = Path("fate_oia/engine/audit_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert 'return ({name: "PASS" for name in _REQUIRED_FUNCTIONAL_CHECKS}' not in source
    assert 'checks["direct_image"] = "PASS"' in source
    assert 'checks["artifact_schema_v4"] = "PASS"' in source
    assert "functional checks lack explicit evidence" in source


def test_real_factor_gate_writer_binds_the_audited_git_head() -> None:
    source = Path("fate_oia/engine/audit_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert 'gate["git_head"] = result["git_head"]' in source


def test_dynamic_visual_credibility_audit_cannot_use_reason_anchor_definitions() -> None:
    source = Path("fate_oia/engine/audit_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    dynamic_call = source.split("if args.real_factor_audit_rows:", 1)[1].split(
        "real_factor_path", 1
    )[0]
    assert 'source_split="audit_visual"' in dynamic_call
    assert "factor_definitions=" not in dynamic_call


class _RealForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_adapter = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.reason_adapter = nn.Conv2d(3, 1, kernel_size=1, bias=False)

    def forward(self, images: torch.Tensor, *, return_masks: bool, **_: object) -> dict[str, torch.Tensor]:
        assert return_masks is True
        action_map = self.action_adapter(images)
        reason_map = self.reason_adapter(images)
        factor_masks = torch.sigmoid(torch.cat((action_map, reason_map), dim=1))
        return {
            "action_final_logits": action_map.mean((2, 3)),
            "action_factor_off_logits": action_map.mean((2, 3)),
            "action_factor_shuffled_logits": action_map.mean((2, 3)) + 0.01,
            "action_wrong_target_logits": action_map.mean((2, 3)) - 0.01,
            "action_equal_mass_random_logits": action_map.mean((2, 3)) + 0.02,
            "reason_observed_logits": reason_map.mean((2, 3)),
            "reason_propensity": torch.full_like(reason_map.mean((2, 3)), 0.5),
            "factor_soft_masks": factor_masks,
            "support_weights": factor_masks.mean((2, 3)).unsqueeze(-1),
            "veto_weights": (1.0 - factor_masks).mean((2, 3)).unsqueeze(-1),
        }


def test_dynamic_audit_uses_real_forwards_and_lane_gradients() -> None:
    model = _RealForwardModel()
    result = verify_dynamic_forward_and_gradients(model, torch.randn(2, 3, 8, 8))

    assert result["pass"] is True
    assert result["forward_calls"] >= 2
    assert result["input_sensitivity"]["action_final_logits"] > 0.0
    assert result["gradient_sum_abs"]["action_adapter"] > 0.0
    assert result["gradient_sum_abs"]["reason_adapter"] > 0.0
    assert result["gradient_firewall"]["reason_to_action_adapter"] == 0.0
    assert result["gradient_firewall"]["action_to_reason_adapter"] == 0.0


def test_dynamic_audit_fails_closed_when_a_required_real_output_is_missing() -> None:
    class MissingEdges(_RealForwardModel):
        def forward(self, images: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            output = super().forward(images, **kwargs)
            output.pop("support_weights")
            return output

    with pytest.raises(ICDORAuditError, match="support_weights"):
        verify_dynamic_forward_and_gradients(MissingEdges(), torch.randn(2, 3, 8, 8))


def test_protocol_hard_gate_binds_current_hashes_and_every_gate(tmp_path) -> None:
    review = {
        "status": "PASS",
        "target_head": "abc",
        "resolved_config_sha256": "CONFIG",
        "runtime_selection_sha256": "RUNTIME",
        "gates": {"runtime_profile": "PASS", "artifact_schema": "PASS"},
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")

    protocol_hard_gate(
        review_path,
        target_head="abc",
        config_sha256="CONFIG",
        runtime_sha256="RUNTIME",
        required_gates=("runtime_profile", "artifact_schema"),
    )

    with pytest.raises(ICDORAuditError, match="config hash"):
        protocol_hard_gate(
            review_path,
            target_head="abc",
            config_sha256="DRIFT",
            runtime_sha256="RUNTIME",
            required_gates=("runtime_profile",),
        )


def test_review_pass_is_fail_closed_and_binds_all_evidence() -> None:
    audit = {
        "pass": True,
        "git_head": "abc",
        "config_sha256": "CONFIG",
        "source": {"pass": True},
        "dynamic_forward": {"pass": True},
        "functional_checks": {name: "PASS" for name in (
            "direct_image", "factor_certificate", "edge_admission", "action_firewall",
            "reason_firewall", "selective_observation", "calibration", "artifact_schema",
            "resume_integrity", "visual_audit", "foreground_launcher", "continuous_credibility",
            "fine_transport", "partial_action_admission", "regime_schedule", "artifact_schema_v4", "target_utility",
            "batch_field_reuse",
        )},
        "missing_items": [],
        "git_tree": "TREE",
        "source_manifest_sha256": "SOURCES",
        "contract_manifest_sha256": "CONTRACT",
        "split_protocol": {"split_sha256": "SPLIT"},
        "worktree_clean": True,
    }
    runtime = {"pass": True, "selected": {"status": "PASS"}}
    pilot = {
        "pass": True, "artifacts_complete": True, "pending_artifacts": [], "git_head": "abc",
        "certificate_sha256": "CERTIFICATE", "edge_admission_sha256": "EDGE",
        "semantic_validation": {"pass": True, "errors": []},
        "deployment_admission_ready": True,
    }

    review = build_review_pass(
        audit, runtime, pilot, runtime_sha256="RUNTIME", pilot_sha256="PILOT",
        evidence_files=_bound_evidence_files(),
        remediation_gates=_passing_remediation_gates(),
        final_remediation_plan_sha256="PLAN",
        audit_addendum_sha256="ADDENDUM",
    )

    assert review["status"] == "PASS"
    assert review["target_head"] == "abc"
    assert review["target_tree"] == "TREE"
    assert review["source_manifest_sha256"] == "SOURCES"
    assert review["contract_manifest_sha256"] == "CONTRACT"
    assert review["final_remediation_plan_sha256"] == "PLAN"
    assert review["audit_addendum_sha256"] == "ADDENDUM"
    assert review["split_sha256"] == "SPLIT"
    assert review["factor_certificate_sha256"] == "CERTIFICATE"
    assert review["edge_admission_sha256"] == "EDGE"
    assert review["pilot_artifact_sha256"] == "PILOT"
    assert review["evidence_files"] == _bound_evidence_files()
    inconsistent = _bound_evidence_files()
    inconsistent["factor_certificate"] = {**inconsistent["factor_certificate"], "sha256": "OTHER"}
    with pytest.raises(ICDORAuditError, match="evidence.*inconsistent"):
        build_review_pass(
            audit, runtime, pilot, runtime_sha256="RUNTIME", pilot_sha256="PILOT",
            evidence_files=inconsistent,
            remediation_gates=_passing_remediation_gates(),
            final_remediation_plan_sha256="PLAN", audit_addendum_sha256="ADDENDUM",
        )
    assert set(review["gates"].values()) == {"PASS"}
    broken = dict(pilot, pending_artifacts=["visual_audit_manifest.json"])
    with pytest.raises(ICDORAuditError, match="pending"):
        build_review_pass(
            audit, runtime, broken, runtime_sha256="RUNTIME", pilot_sha256="PILOT",
            evidence_files=_bound_evidence_files(),
            remediation_gates=_passing_remediation_gates(),
            final_remediation_plan_sha256="PLAN",
            audit_addendum_sha256="ADDENDUM",
        )

    stale = _passing_remediation_gates()
    stale["REAL_FACTOR_AUDIT"] = {**stale["REAL_FACTOR_AUDIT"], "git_head": "stale"}
    with pytest.raises(ICDORAuditError, match="current audited HEAD"):
        build_review_pass(
            audit, runtime, pilot, runtime_sha256="RUNTIME", pilot_sha256="PILOT",
            evidence_files=_bound_evidence_files(),
            remediation_gates=stale,
            final_remediation_plan_sha256="PLAN",
            audit_addendum_sha256="ADDENDUM",
        )

    with pytest.raises(ICDORAuditError, match="pilot gate.*HEAD"):
        build_review_pass(
            audit, runtime, {**pilot, "git_head": "stale"}, runtime_sha256="RUNTIME", pilot_sha256="PILOT",
            evidence_files=_bound_evidence_files(),
            remediation_gates=_passing_remediation_gates(),
            final_remediation_plan_sha256="PLAN",
            audit_addendum_sha256="ADDENDUM",
        )

    with pytest.raises(ICDORAuditError, match="pilot evidence bindings"):
        build_review_pass(
            audit, runtime, {**pilot, "certificate_sha256": None},
            runtime_sha256="RUNTIME", pilot_sha256="PILOT", evidence_files=_bound_evidence_files(),
            remediation_gates=_passing_remediation_gates(),
            final_remediation_plan_sha256="PLAN",
            audit_addendum_sha256="ADDENDUM",
        )

    with pytest.raises(ICDORAuditError, match="pilot semantic validation"):
        build_review_pass(
            audit, runtime, {**pilot, "semantic_validation": {"pass": False}},
            runtime_sha256="RUNTIME", pilot_sha256="PILOT", evidence_files=_bound_evidence_files(),
            remediation_gates=_passing_remediation_gates(),
            final_remediation_plan_sha256="PLAN",
            audit_addendum_sha256="ADDENDUM",
        )


def test_learning_access_review_does_not_claim_or_require_deployment_admission() -> None:
    audit = {
        "pass": True, "git_head": "abc", "git_tree": "TREE",
        "source_manifest_sha256": "SOURCES", "contract_manifest_sha256": "CONTRACT",
        "worktree_clean": True, "split_protocol": {"split_sha256": "SPLIT"},
        "config_sha256": "CONFIG",
        "functional_checks": {name: "PASS" for name in (
            "direct_image", "factor_certificate", "edge_admission", "action_firewall",
            "reason_firewall", "selective_observation", "calibration", "artifact_schema",
            "resume_integrity", "visual_audit", "foreground_launcher", "continuous_credibility",
            "fine_transport", "partial_action_admission", "regime_schedule", "artifact_schema_v4",
            "target_utility", "batch_field_reuse",
        )},
        "missing_items": [],
    }
    pilot = {
        "pass": True, "artifacts_complete": True, "pending_artifacts": [], "git_head": "abc",
        "certificate_sha256": "CERTIFICATE", "edge_admission_sha256": "EDGE",
        "semantic_validation": {"pass": True, "errors": []},
        "deployment_admission_ready": False,
    }
    review = build_review_pass(
        audit, {"pass": True, "selected": {"status": "PASS"}}, pilot,
        runtime_sha256="RUNTIME", pilot_sha256="PILOT", evidence_files=_bound_evidence_files(),
        remediation_gates=_passing_remediation_gates(),
        final_remediation_plan_sha256="PLAN", audit_addendum_sha256="ADDENDUM",
    )
    assert review["status"] == "PASS"
    assert review["review_scope"] == "learning_access_preflight"
    assert review["deployment_admission_ready"] is False
    assert review["final_deployment_claim_allowed"] is False


def test_review_pass_rejects_dirty_or_unbound_source_tree() -> None:
    checks = (
        "direct_image", "factor_certificate", "edge_admission", "action_firewall",
        "reason_firewall", "selective_observation", "calibration", "artifact_schema",
        "resume_integrity", "visual_audit", "foreground_launcher", "continuous_credibility",
        "fine_transport", "partial_action_admission", "regime_schedule", "artifact_schema_v4",
        "batch_field_reuse",
    )
    base = {
        "pass": True, "git_head": "abc", "git_tree": "TREE",
        "source_manifest_sha256": "SOURCES", "contract_manifest_sha256": "CONTRACT",
        "worktree_clean": True, "split_protocol": {"split_sha256": "SPLIT"},
        "config_sha256": "CONFIG", "functional_checks": {name: "PASS" for name in checks},
        "missing_items": [],
    }
    runtime = {"pass": True, "selected": {"status": "PASS"}}
    pilot = {
        "pass": True, "artifacts_complete": True, "pending_artifacts": [], "git_head": "abc",
        "certificate_sha256": "CERTIFICATE", "edge_admission_sha256": "EDGE",
        "semantic_validation": {"pass": True, "errors": []},
    }
    for mutation in ({"worktree_clean": False}, {"git_tree": ""}, {"source_manifest_sha256": ""}, {"contract_manifest_sha256": ""}):
        with pytest.raises(ICDORAuditError, match="source tree"):
            build_review_pass(
                dict(base, **mutation), runtime, pilot, runtime_sha256="RUNTIME", pilot_sha256="PILOT",
                evidence_files=_bound_evidence_files(),
                remediation_gates=_passing_remediation_gates(),
                final_remediation_plan_sha256="PLAN",
                audit_addendum_sha256="ADDENDUM",
            )

