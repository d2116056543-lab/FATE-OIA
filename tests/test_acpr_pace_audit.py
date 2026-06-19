import json
import subprocess
import sys


def test_pace_audit_module_writes_review_pass(tmp_path):
    out = tmp_path / "audit"
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.audit_acpr_pace_implementation",
        "--config",
        "configs/fate_oia_train_360x640_acpr_pace_v1.yaml",
        "--output_dir",
        str(out),
        "--device",
        "cpu",
        "--write_review_pass",
    ]
    subprocess.check_call(cmd)
    data = json.loads((out / "implementation_audit_ACPR_PACE_V1.json").read_text())
    assert data["pass"] is True
    assert (out / "REVIEW_PASS_ACPR_PACE_V1.txt").exists()
