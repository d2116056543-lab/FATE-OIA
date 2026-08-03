import json
from pathlib import Path

import pytest

from fate_oia.engine.supervise_save_oia_foreground import (
    assert_foreground_command,
    validate_save_full_readiness,
)


def test_supervisor_rejects_background_mechanisms():
    for token in ("Start-Process", "Start-Job", "nohup", "--daemon"):
        with pytest.raises(ValueError):
            assert_foreground_command(["python", token])


def test_readiness_requires_same_complete_binding_chain(tmp_path: Path):
    bindings = _bindings()
    review = {"pass": True, "bindings": bindings}
    pilot = {"pass": True, "bindings": bindings, "gates": {x: True for x in "ABCDEFG"}}
    profile = {"pass": True, "bindings": bindings, "chosen": {"batch_size": 4, "gradient_accumulation_steps": 8}}
    ready = {"pass": True, "bindings": bindings}
    paths = []
    for name, payload in (("review", review), ("pilot", pilot), ("profile", profile), ("ready", ready)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    result = validate_save_full_readiness(*paths, expected_git_head="head")
    assert result["batch_size"] == 4
    pilot["bindings"] = dict(bindings, schema_hash="different")
    paths[1].write_text(json.dumps(pilot), encoding="utf-8")
    with pytest.raises(RuntimeError):
        validate_save_full_readiness(*paths, expected_git_head="head")


def test_review_pending_runtime_hashes_do_not_block_a_bound_pilot(tmp_path: Path):
    bindings = _bindings()
    review_bindings = dict(bindings, checkpoint_hash="pending", logits_hash="pending", labels_hash="pending", file_order_hash="pending", split_hash="pending")
    review = {"pass": True, "bindings": review_bindings}
    pilot = {"pass": True, "bindings": bindings, "gates": {x: True for x in "ABCDEFG"}}
    profile = {"pass": True, "bindings": review_bindings, "chosen": {"batch_size": 4, "gradient_accumulation_steps": 8}}
    ready = {"pass": True, "bindings": bindings}
    paths = []
    for name, payload in (("review", review), ("pilot", pilot), ("profile", profile), ("ready", ready)):
        path = tmp_path / f"{name}.json"; path.write_text(json.dumps(payload), encoding="utf-8"); paths.append(path)
    assert validate_save_full_readiness(*paths, expected_git_head="head")["bindings"] == bindings


def test_supervisor_allows_only_explicit_safe_numeric_candidate(tmp_path: Path):
    bindings = _bindings()
    review = {"pass": True, "bindings": bindings}
    profile = {"pass": True, "bindings": bindings, "chosen": {"batch_size": 4, "gradient_accumulation_steps": 8}}
    pilot = {"pass": False, "numeric_candidate_eligible": True, "bindings": bindings, "gates": {x: x in "AG" for x in "ABCDEFG"}}
    ready = {"pass": False, "numeric_candidate_eligible": True, "bindings": bindings}
    paths = []
    for name, payload in (("review", review), ("pilot", pilot), ("profile", profile), ("ready", ready)):
        path = tmp_path / f"{name}.json"; path.write_text(json.dumps(payload), encoding="utf-8"); paths.append(path)
    with pytest.raises(RuntimeError):
        validate_save_full_readiness(*paths, expected_git_head="head")
    result = validate_save_full_readiness(*paths, expected_git_head="head", allow_numeric_candidate=True)
    assert result["selection_mode"] == "safe_numeric_candidate"


def _bindings():
    values = {name: name for name in (
        "config_hash", "source_tree_hash", "schema_hash", "split_hash",
        "checkpoint_hash", "logits_hash", "labels_hash", "file_order_hash",
    )}
    values["git_head"] = "head"
    return values
