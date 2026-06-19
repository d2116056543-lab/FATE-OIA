import json
import subprocess
import sys


def test_visual_export_requires_real_contribution_artifact(tmp_path):
    epoch = tmp_path / "epoch_000"
    epoch.mkdir()
    out = tmp_path / "visuals"
    subprocess.check_call([
        sys.executable,
        "-m",
        "fate_oia.engine.export_acpr_pace_visuals",
        "--epoch_dir",
        str(epoch),
        "--output_dir",
        str(out),
    ])
    payload = json.loads((out / "pace_visual_manifest.json").read_text(encoding="utf-8"))
    assert payload["available"] is False
    assert "missing" in payload["reason"]
