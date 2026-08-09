import json

import pytest

from fate_oia.engine.evaluate_pact_oia_probe import validate_external_bindings


def test_external_artifacts_must_bind_current_code_config_and_checkpoint(tmp_path):
    audit = tmp_path / "audit.json"
    selected = tmp_path / "selected.json"
    audit.write_text(json.dumps({
        "pass": True,
        "git_head": "head",
        "config_hash": "config",
        "checkpoint_hash": "checkpoint",
    }), encoding="utf-8")
    selected.write_text(json.dumps({"action_scale": 1.0}), encoding="utf-8")
    result, hparams = validate_external_bindings(
        audit, selected, "head", "config", "checkpoint")
    assert result["pass"] and hparams["action_scale"] == 1.0
    with pytest.raises(RuntimeError, match="git_head mismatch"):
        validate_external_bindings(audit, selected, "different", "config", "checkpoint")
