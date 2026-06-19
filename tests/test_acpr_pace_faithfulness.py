import json
import subprocess
import sys


def test_faithfulness_eval_is_eval_only_and_reports_missing_artifacts(tmp_path):
    out = tmp_path / "faith"
    subprocess.check_call([sys.executable, "-m", "fate_oia.engine.eval_acpr_pace_faithfulness", "--output_dir", str(out)])
    payload = json.loads((out / "pace_faithfulness_eval.json").read_text(encoding="utf-8"))
    assert payload["eval_only"] is True
    assert payload["optimizer_update"] is False
