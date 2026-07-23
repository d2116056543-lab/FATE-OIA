from pathlib import Path

from fate_oia.engine.audit_precise_oia_implementation import REQUIRED, _scan_forbidden, REQUIRED_PILOT_CHECKS


def test_audit_requires_full_precise_source_surface():
    assert len(REQUIRED) >= 30
    assert "fate_oia/models/precise_oia_model.py" in REQUIRED


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


def test_full_gate_binds_pilot_checkpoint_and_raw_artifacts():
    audit = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "audit_precise_oia_implementation.py").read_text(encoding="utf-8")
    supervisor = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "supervise_precise_oia_foreground.py").read_text(encoding="utf-8")
    for token in ("pilot_artifact_hashes", "checkpoint_latest.pth", "pcvl_probabilities.pt", "test_file_names_sha256"):
        assert token in audit
    assert "pilot_artifact_hashes" in supervisor


def test_audit_accepts_skill_post_pilot_mode_as_full_gate_mode():
    source = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "audit_precise_oia_implementation.py").read_text(encoding="utf-8")
    assert 'mode in {"pilot", "post_pilot"}' in source


def test_epoch_artifact_gate_loads_and_validates_tensor_contents():
    source = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "audit_precise_oia_implementation.py").read_text(encoding="utf-8")
    for token in ("torch.load", "torch.isfinite", "expected_samples", "file_names.json"):
        assert token in source
