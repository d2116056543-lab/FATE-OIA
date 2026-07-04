import json
import subprocess
import sys
from pathlib import Path


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
    assert review["no_reason_to_final_action"] is True
    assert review["no_raw_qrho_to_action_delta"] is True
    assert review["best_action_and_exp_checkpoints"] is True
    assert review["pareto_gradient_stats_dynamic_firewall"] is True
    assert review["branch_ablation_not_stub"] is True
