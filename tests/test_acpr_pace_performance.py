import json
import subprocess
import sys


def test_audit_writes_performance_audit(tmp_path):
    out = tmp_path / "audit"
    subprocess.check_call([
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
    ])
    payload = json.loads((out / "implementation_audit_ACPR_PACE_V1.json").read_text(encoding="utf-8"))
    perf = json.loads((out / "performance_audit.json").read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert perf["pass"] is True
    assert "forward_overhead_ratio" in perf
