from __future__ import annotations

from pathlib import Path

from fate_oia.engine.audit_care_act_oia_implementation import run_static_audit


def test_static_audit_checks_required_files_and_forbidden_foreground_tokens():
    root = Path.cwd()
    result = run_static_audit(root, config_path=root / "configs" / "fate_oia_train_360x640_care_act_oia_v1.yaml")
    assert result["review"] in {"PASS", "FAIL"}
    assert "checks" in result
    assert any(x["name"] == "foreground_no_detach_tokens" for x in result["checks"])
