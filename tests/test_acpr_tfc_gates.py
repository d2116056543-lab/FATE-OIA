import json
import subprocess
import sys
from pathlib import Path

from fate_oia.models.tfc_action_head import action_delta_cap
from fate_oia.models.tfc_reason_head import reason_delta_cap


def test_tfc_delta_schedule_matches_plan():
    assert [round(action_delta_cap(e), 4) for e in range(0, 12)] == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.04
    ]
    assert [round(reason_delta_cap(e), 4) for e in range(0, 12)] == [
        0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.08, 0.09, 0.10, 0.11, 0.12, 0.10
    ]


def test_tfc_audit_gates_smoke(tmp_path):
    out = tmp_path / "review"
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.audit_tfc_gates",
        "--config",
        "configs/fate_oia_train_360x640_acpr_tfc_v1.yaml",
        "--mode",
        "all",
        "--device",
        "cpu",
        "--batch_size",
        "1",
        "--output_dir",
        str(out),
        "--write_review_pass",
    ]
    subprocess.check_call(cmd)
    review = json.loads(Path(".review/acpr_tfc_v1_REVIEW_PASS.json").read_text())
    assert review["review_pass"] is True
    assert review["gate_count"] >= 7
    assert all(review["gate_passes"])
    assert review["forward_schema_pass"] is True
    assert review["memory_probe_pass"] is True
    assert review["no_reason_to_final_action"] is True
    assert review["no_raw_qrho_to_action_delta"] is True
    assert review["best_action_and_exp_checkpoints"] is True
    assert review["pareto_gradient_stats_dynamic_firewall"] is True
    assert review["branch_ablation_not_stub"] is True
    assert review["pretrain_gates_required_by_default"] is True
    assert review["allow_failed_gates_used"] is True
    assert review["oracle_act_drop_stop_condition"] is True
    assert review["map_threshold_movement_stop_condition"] is True
    assert review["foreground_script_argparse_safe_review_flag"] is True
    assert review["train_checks_gate_json_pass_values"] is True
    assert review["audit_exits_nonzero_on_failed_review"] is True
    assert review["target_credit_stats_written_every_epoch"] is True
    assert review["delta_schedule_matches_plan"] is True
    assert review["scheduler_and_lr_groups_used"] is True
    assert review["factor_bank_target_indices_range_checked"] is True
    assert review["target_credit_masks_unknown_native_zero"] is True
    assert review["pu_hard_negative_requires_deletion_gate"] is True
    assert review["reason_delta_requires_deletion_mask"] is True
    assert review["reason_deletion_stats_written"] is True
    assert review["factor_measurement_lr_group_names_correct"] is True
    assert review["pareto_optimizer_functional"] is True
