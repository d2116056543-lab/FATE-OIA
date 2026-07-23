from pathlib import Path

from fate_oia.engine.audit_precise_oia_implementation import (
    REQUIRED,
    REQUIRED_CURRICULUM_CHECKS,
    REQUIRED_PILOT_CHECKS,
    _scan_forbidden,
)

ROOT = Path(__file__).resolve().parents[1]


def test_audit_requires_full_precise_source_surface():
    assert len(REQUIRED) >= 30
    assert "fate_oia/models/precise_oia_model.py" in REQUIRED
    assert "fate_oia/engine/precise_curriculum.py" in REQUIRED


def test_curriculum_gate_covers_static_schedule_owner_and_override_contracts():
    assert {
        "preflight_gate_current",
        "fixed_schedule_exact",
        "owner_mapping_exact",
        "owner_active_epoch_totals_exact",
        "inactive_optimizer_state_empty_at_launch",
        "threshold_deploy_scaled",
        "runtime_assertions_wired",
        "runtime_profile_bound",
        "embedded_curriculum_override_bound",
        "old_full_gate_not_authoritative",
        "skill_authorizes_embedded_curriculum",
        "trainer_enforces_curriculum_gate",
        "inactive_intervention_skipped",
        "owner_local_lifecycle_asserted",
        "strict_resume_lifecycle",
        "threshold_teacher_full_activation_only",
    } == REQUIRED_CURRICULUM_CHECKS


def test_audit_exports_both_review_and_training_source_hashes():
    source = (
        ROOT / "fate_oia" / "engine" / "audit_precise_oia_implementation.py"
    ).read_text(encoding="utf-8")
    assert '"source_tree_sha256": source_tree_sha' in source
    assert '"training_source_sha256": _training_source_sha(root)' in source


def test_forbidden_scan_covers_config_and_script_without_self_matching_audit_rules():
    root = Path(__file__).resolve().parents[1]
    forbidden, incomplete = _scan_forbidden(root)
    assert all("audit_precise_oia_implementation.py" not in paths for paths in (*forbidden.values(), *incomplete.values()))
    assert "Start-Process" not in forbidden


def test_pilot_gate_covers_every_plan_defined_runtime_and_mechanism_requirement():
    assert {
        "pilot_identity_matches_current_code",
        "pilot_sample_contract",
        "three_epochs_complete",
        "mechanism_epoch_coverage",
        "epoch_artifacts_complete",
        "dino_call_count_one",
        "peak_reserved_under_hard_limit",
        "all_intended_owners_stepped",
        "pcvl_optimizer_stepped",
        "observed_firewall_exact_zero",
        "action_exchange_ratio_in_range",
        "reason_exchange_ratio_in_range",
        "action_reread_ratio_in_range",
        "reason_reread_ratio_in_range",
        "reliability_noncollapsed",
        "reference_not_center_collapsed",
        "selected_beats_control",
        "selected_beats_control_action",
        "selected_beats_control_reason",
        "target_counterfactual_coverage",
        "evidence_shuffle_changes_reason",
        "annotation_delta_nonzero",
        "pcvl_artifacts_complete",
        "pcvl_predicate_action_value_supported",
        "pcvl_learned_evidence_supported",
        "pcvl_learned_exchange_supported",
    }.issubset(REQUIRED_PILOT_CHECKS)


def test_selected_control_gate_reads_heldout_epoch_metrics_not_train_batches():
    source = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "audit_precise_oia_implementation.py").read_text(encoding="utf-8")
    assert 'latest_metrics.get("counterfactual", {})' in source
    assert 'heldout_counterfactual.get("selected_control_margin"' in source
    assert '"selected_beats_control_action"' in source
    assert '"selected_beats_control_reason"' in source
    assert 'positive_rate' in source and 'valid_targets' in source and 'dominance' in source
    assert 'logits_reason_evidence_shuffled.pt' in source


def test_full_gate_binds_pilot_checkpoint_and_raw_artifacts():
    audit = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "audit_precise_oia_implementation.py").read_text(encoding="utf-8")
    supervisor = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "supervise_precise_oia_foreground.py").read_text(encoding="utf-8")
    for token in ("pilot_artifact_hashes", "checkpoint_latest.pth", "pcvl_probabilities.pt", "test_file_names_sha256"):
        assert token in audit
    assert "pilot_artifact_hashes" in supervisor


def test_audit_accepts_skill_post_pilot_mode_as_full_gate_mode():
    source = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "audit_precise_oia_implementation.py").read_text(encoding="utf-8")
    assert 'mode in {"pilot", "post_pilot"}' in source


def test_audit_has_a_distinct_full_curriculum_gate_mode():
    source = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "audit_precise_oia_implementation.py").read_text(encoding="utf-8")
    assert "FULL_CURRICULUM_READY" in source
    assert "PRECISE_OIA_V1_FULL_CURRICULUM_READY.json" in source
    assert "--write_full_curriculum_ready" in source


def test_epoch_artifact_gate_loads_and_validates_tensor_contents():
    source = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "audit_precise_oia_implementation.py").read_text(encoding="utf-8")
    for token in ("torch.load", "torch.isfinite", "expected_samples", "file_names.json"):
        assert token in source
