import subprocess
import sys


def test_fusionlite_audit_cpu(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.audit_acpr_fusionlite_implementation",
        "--config",
        "configs/fate_oia_train_360x640_acpr_fusionlite_v1_4.yaml",
        "--output_dir",
        str(tmp_path),
        "--device",
        "cpu",
        "--write_review_pass",
    ]
    subprocess.check_call(cmd)
    assert (tmp_path / "REVIEW_PASS_ACPR_FUSIONLITE_V1_4.txt").exists()
