"""P21 RED contracts for fail-closed RAEL implementation auditing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == "tests" else HERE.parents[2]
STAGING = ROOT / "remote_patch" / "P21"
AUDIT = STAGING / "audit_acpr_rael_oia.py" if STAGING.is_dir() else ROOT / "fate_oia" / "engine" / "audit_acpr_rael_oia.py"


def _module():
    spec = importlib.util.spec_from_file_location("rael_p21_audit", AUDIT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_has_fail_closed_required_file_and_forbidden_checks() -> None:
    module = _module()
    assert module.REQUIRED_SOURCE_FILES
    assert module.REQUIRED_TEST_FILES
    assert module.FORBIDDEN_PATTERNS
    assert callable(module.run_audit)
    assert callable(module.validate_smoke_artifacts)
    # P21 must bind the complete plan-level file/test inventory, not merely
    # the five files introduced by P21.
    assert "fate_oia/models/rael_oia_model.py" in module.REQUIRED_SOURCE_FILES
    assert "fate_oia/losses/rael_counterfactual_losses.py" in module.REQUIRED_SOURCE_FILES
    assert "tests/test_rael_counterfactual.py" in module.REQUIRED_TEST_FILES
    assert "tests/test_rael_worktree_contract.py" in module.REQUIRED_TEST_FILES
    assert "tests/test_rael_trainer_handoff.py" in module.REQUIRED_TEST_FILES


def test_audit_requires_explicit_user_pilot_override_not_fake_pilot(tmp_path: Path) -> None:
    module = _module()
    payload = module.pilot_override_payload(
        enabled=True,
        reason="user requested one minimal smoke before full train",
    )
    assert payload["pilot_protocol_override"] is True
    assert payload["pilot_completed"] is False
    with pytest.raises(ValueError):
        module.pilot_override_payload(enabled=False, reason="")


def test_smoke_validator_rejects_placeholder_and_invalid_artifacts(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "mechanism_stats.jsonl").write_text(json.dumps({"placeholder": True}) + "\n", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        module.validate_smoke_artifacts(tmp_path)


def test_smoke_validator_binds_current_head_config_and_source(tmp_path: Path) -> None:
    module = _module()
    head = "d" * 40
    config_sha = "e" * 64
    source = {"required_files_hash": "f" * 64, "complete": True}
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"mode": "smoke", "synthetic": False, "git_head": head, "config_sha256": config_sha}),
        encoding="utf-8",
    )
    (tmp_path / "config_resolved.yaml").write_text("training: {}\n", encoding="utf-8")
    (tmp_path / "source_fingerprint.json").write_text(json.dumps(source), encoding="utf-8")
    (tmp_path / "mechanism_stats.jsonl").write_text(
        json.dumps({"dino_call_count": 1, "optimizer_step": 1, "finite": True, "owner_parameter_delta": {"x": 0.1}}) + "\n",
        encoding="utf-8",
    )
    result = module.validate_smoke_artifacts(
        tmp_path,
        expected_git_head=head,
        expected_config_sha256=config_sha,
        expected_source_fingerprint=source,
    )
    assert result["passed"] is True
    with pytest.raises(ValueError):
        module.validate_smoke_artifacts(
            tmp_path,
            expected_git_head="0" * 40,
            expected_config_sha256=config_sha,
            expected_source_fingerprint=source,
        )


def test_audit_record_binds_git_head_and_does_not_write_pass_when_failed(tmp_path: Path) -> None:
    module = _module()
    record = module.build_audit_record(
        git_head="b" * 40,
        required_files={"x.py": False},
        required_tests={"x_test.py": False},
        forbidden_results={"forbidden": []},
        compile_result={"passed": False},
        pytest_result={"passed": False},
        smoke_result={"passed": False},
        pilot_override=module.pilot_override_payload(enabled=True, reason="user override"),
    )
    assert record["pass"] is False
    assert record["git_head"] == "b" * 40
    with pytest.raises(ValueError):
        module.write_review_pass(tmp_path, record)
    with pytest.raises(ValueError):
        module.write_full_train_gate(tmp_path, record)


def test_full_train_gate_binds_smoke_override_and_clean_audit(tmp_path: Path) -> None:
    module = _module()
    record = module.build_audit_record(
        git_head="c" * 40,
        required_files={"x.py": True},
        required_tests={"x_test.py": True},
        forbidden_results={"forbidden": []},
        compile_result={"passed": True},
        pytest_result={"passed": True},
        smoke_result={"passed": True, "row_count": 3},
        pilot_override=module.pilot_override_payload(
            enabled=True,
            reason="user requested one minimal real smoke before full train",
        ),
        grounding_layout={"passed": True},
    )
    assert record["pass"] is True
    gate = module.write_full_train_gate(tmp_path, record)
    payload = json.loads(gate.read_text(encoding="utf-8"))
    assert gate.name == "RAEL_OIA_V1_FULL_TRAIN_READY.json"
    assert payload["pass"] is True
    assert payload["git_head"] == "c" * 40
    assert payload["pilot_override"]["replacement"] == "minimal_real_smoke_only"
    assert payload["smoke_result"]["passed"] is True
    assert payload["unresolved"] == []


def test_audit_never_claims_an_unrun_pilot() -> None:
    module = _module()
    payload = module.pilot_override_payload(enabled=True, reason="user explicitly requested minimal smoke")
    assert payload == {
        "pilot_protocol_override": True,
        "pilot_completed": False,
        "reason": "user explicitly requested minimal smoke",
        "replacement": "minimal_real_smoke_only",
    }


def test_audit_requires_explicit_existing_per_frame_grounding_directories(tmp_path: Path) -> None:
    module = _module()
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    (train / "a.json").write_text("{}", encoding="utf-8")
    (val / "b.json").write_text("{}", encoding="utf-8")
    result = module.validate_explicit_grounding_layout(
        {"grounding_sources": {"label_directories": {"train": str(train), "val": str(val)}}}
    )
    assert result["passed"] is True
    with pytest.raises((FileNotFoundError, ValueError)):
        module.validate_explicit_grounding_layout({"grounding_sources": {"label_directories": {"train": str(train)}}})
