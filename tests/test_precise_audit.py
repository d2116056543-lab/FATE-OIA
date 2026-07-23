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
    }.issubset(REQUIRED_PILOT_CHECKS)
