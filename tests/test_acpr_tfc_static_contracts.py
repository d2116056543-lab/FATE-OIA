from pathlib import Path


def test_tfc_branch_ablation_exports_full_action_diagnostics():
    src = Path("fate_oia/engine/eval_tfc_branch_ablation.py").read_text(encoding="utf-8")
    required_markers = [
        "action_threshold_delta_off",
        "per_action_AP_AUC_F1",
        "FP_to_TP",
        "TP_to_FN",
        "TN_to_FP",
        "FN_to_TN",
        "deploy_vs_base_delta",
    ]
    missing = [marker for marker in required_markers if marker not in src]
    assert not missing, f"missing TFC branch ablation diagnostics: {missing}"


def test_tfc_audit_hard_gates_branch_ablation_diagnostics():
    src = Path("fate_oia/engine/audit_tfc_gates.py").read_text(encoding="utf-8")
    required_markers = [
        "branch_ablation_full_diagnostics",
        "action_threshold_delta_off",
        "deploy_vs_base_delta",
        "FP_to_TP",
        "TN_to_FP",
    ]
    missing = [marker for marker in required_markers if marker not in src]
    assert not missing, f"missing TFC audit hard gate markers: {missing}"


def test_tfc_run_manifest_records_reproducibility_fields():
    train_src = Path("fate_oia/engine/train_acpr_tfc_oia.py").read_text(encoding="utf-8")
    audit_src = Path("fate_oia/engine/audit_tfc_gates.py").read_text(encoding="utf-8")
    required_train_markers = [
        "\"git_head\"",
        "\"command_line\"",
        "\"pretrained_weights\"",
        "\"selected_layers\"",
        "\"best_selection\"",
        "\"foreground_only\"",
        "\"require_review_pass\"",
    ]
    missing_train = [marker for marker in required_train_markers if marker not in train_src]
    assert not missing_train, f"missing run_manifest reproducibility fields: {missing_train}"
    assert "run_manifest_records_reproducibility_fields" in audit_src


def test_tfc_static_contracts_are_part_of_required_audit_suite():
    audit_src = Path("fate_oia/engine/audit_tfc_gates.py").read_text(encoding="utf-8")
    skill_src = Path(".codex/skills/acpr-tfc-implementation-audit/SKILL.md").read_text(encoding="utf-8")
    marker = "tests/test_acpr_tfc_static_contracts.py"
    assert marker in audit_src
    assert marker in skill_src


def test_tfc_review_pass_requires_complete_gate_set():
    src = Path("fate_oia/engine/audit_tfc_gates.py").read_text(encoding="utf-8")
    required_markers = [
        "required_review_gate_names",
        "missing_review_gates",
        "TFC_GATE_A_CODE_AUDIT_PASS.json",
        "TFC_GATE_H_MEMORY_PROBE_PASS.json",
    ]
    missing = [marker for marker in required_markers if marker not in src]
    assert not missing, f"review pass can be written without explicit full gate set check: {missing}"
