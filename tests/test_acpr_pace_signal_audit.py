import json
import subprocess
import sys
from pathlib import Path


def test_signal_audit_writes_required_pass_files(tmp_path):
    out = tmp_path / "signal"
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.audit_acpr_pace_signal",
        "--config",
        "configs/fate_oia_train_360x640_acpr_pace_v1.yaml",
        "--checkpoint",
        "dummy_checkpoint_for_unit_test.pth",
        "--output_dir",
        str(out),
        "--device",
        "cpu",
        "--strengths",
        "0.0",
        "0.5",
    ]
    subprocess.check_call(cmd)
    payload = json.loads((out / "signal_audit_ACPR_PACE_V1.json").read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert (out / "PACE_SIGNAL_PASS.json").exists()
    assert (out / "pace_selected_strength.json").exists()
    assert payload["selected_strength"] in [0.0, 0.5]
