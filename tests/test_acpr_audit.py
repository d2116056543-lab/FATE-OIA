import subprocess
import sys


def test_acpr_audit_cpu_pass(tmp_path):
    cmd = [
        sys.executable, "-m", "fate_oia.engine.audit_acpr_oia_implementation",
        "--config", "configs/fate_oia_train_360x640_acpr_oia_v1.yaml",
        "--output_dir", str(tmp_path),
        "--device", "cpu",
        "--write_review_pass",
    ]
    subprocess.check_call(cmd)
    assert (tmp_path / "REVIEW_PASS_ACPR_OIA_V1.txt").exists()
