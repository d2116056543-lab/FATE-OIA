from pathlib import Path
import hashlib
import json
import inspect

import pytest
import yaml

from fate_oia.engine import train_acpr_meter_oia as trainer
from fate_oia.utils.meter_artifacts import combined_file_hash, python_source_tree_hash


def test_training_protocol_is_test_only_and_has_disjoint_calibration() -> None:
    config = yaml.safe_load(Path("configs/fate_oia_train_360x640_acpr_meter_oia_v1.yaml").read_text(encoding="utf-8"))
    assert config["runtime"]["test_only"] is True
    assert config["posthoc_calibration"]["fit_split"] == "train_calib"
    assert config["best_selection_split"] == "test"
    assert config["splits"]["main_audit_calib_disjoint"] is True


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ready_fixture(tmp_path: Path, *, epochs: int, head: str = "head-a") -> tuple[Path, Path]:
    config_dir = tmp_path / "configs"
    review_dir = tmp_path / ".review"
    config_dir.mkdir()
    review_dir.mkdir()
    source_dir = tmp_path / "fate_oia"
    source_dir.mkdir()
    (source_dir / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    config_path = config_dir / "meter.yaml"
    config_path.write_text("training:\n  epochs: 12\n", encoding="utf-8")
    factor = "factor-schema\n"
    grounding = "grounding-schema\n"
    (config_dir / "meter_factor_schema.yaml").write_text(factor, encoding="utf-8")
    (config_dir / "meter_grounding_schema.yaml").write_text(grounding, encoding="utf-8")
    is_pilot = epochs <= 3
    name = (
        "METER_OIA_V1_PRE_PILOT_READY.json"
        if is_pilot
        else "METER_OIA_V1_FULL_TRAIN_READY.json"
    )
    payload = {
        "artifact": (
            "METER_OIA_V1_PRE_PILOT_READY"
            if is_pilot
            else "METER_OIA_V1_FULL_TRAIN_READY"
        ),
        "HEAD": head,
        "config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "schema_hash": combined_file_hash(
            config_dir / "meter_factor_schema.yaml",
            config_dir / "meter_grounding_schema.yaml",
        ),
        "source_tree_hash": python_source_tree_hash(tmp_path),
        "branch": "test-branch",
        "unresolved": [],
        "real_dino": {"executed": True, "pass": True},
    }
    if not is_pilot:
        payload["pass"] = True
        payload["github_head"] = head
        payload["checks"] = {
            key: True for key in trainer.FULL_READY_REQUIRED_CHECKS
        }
    (review_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    return config_path, review_dir / name


def test_readiness_rejects_stale_head_config_and_schema(tmp_path: Path) -> None:
    config_path, ready_path = _ready_fixture(tmp_path, epochs=3)
    trainer.validate_training_readiness(
        root=tmp_path,
        config_path=config_path,
        epochs=3,
        use_mock_dino=False,
        git_head="head-a",
        git_branch="test-branch",
        remote_head="head-a",
        clean_status="",
        source_tree_hash=python_source_tree_hash(tmp_path),
    )
    payload = json.loads(ready_path.read_text(encoding="utf-8"))
    payload["HEAD"] = "stale-head"
    ready_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="HEAD"):
        trainer.validate_training_readiness(
            root=tmp_path,
            config_path=config_path,
            epochs=3,
            use_mock_dino=False,
            git_head="head-a",
            git_branch="test-branch",
            remote_head="head-a",
            clean_status="",
            source_tree_hash=python_source_tree_hash(tmp_path),
        )
    payload["HEAD"] = "head-a"
    payload["config_hash"] = "stale-config"
    ready_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="config_hash"):
        trainer.validate_training_readiness(
            root=tmp_path,
            config_path=config_path,
            epochs=3,
            use_mock_dino=False,
            git_head="head-a",
            git_branch="test-branch",
            remote_head="head-a",
            clean_status="",
            source_tree_hash=python_source_tree_hash(tmp_path),
        )


def test_full_readiness_rejects_mock_dino_and_failed_gate(tmp_path: Path) -> None:
    config_path, ready_path = _ready_fixture(tmp_path, epochs=12)
    with pytest.raises(RuntimeError, match="mock DINO"):
        trainer.validate_training_readiness(
            root=tmp_path,
            config_path=config_path,
            epochs=12,
            use_mock_dino=True,
            git_head="head-a",
            git_branch="test-branch",
            remote_head="head-a",
            clean_status="",
            source_tree_hash=python_source_tree_hash(tmp_path),
        )
    payload = json.loads(ready_path.read_text(encoding="utf-8"))
    payload["pass"] = False
    ready_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pass=true"):
        trainer.validate_training_readiness(
            root=tmp_path,
            config_path=config_path,
            epochs=12,
            use_mock_dino=False,
            git_head="head-a",
            git_branch="test-branch",
            remote_head="head-a",
            clean_status="",
            source_tree_hash=python_source_tree_hash(tmp_path),
        )


def test_readiness_rejects_dirty_tree_and_incomplete_full_checks(tmp_path: Path) -> None:
    config_path, ready_path = _ready_fixture(tmp_path, epochs=12)
    with pytest.raises(RuntimeError, match="clean_worktree"):
        trainer.validate_training_readiness(
            root=tmp_path,
            config_path=config_path,
            epochs=12,
            use_mock_dino=False,
            git_head="head-a",
            git_branch="test-branch",
            remote_head="head-a",
            clean_status=" M modified.py",
            source_tree_hash=python_source_tree_hash(tmp_path),
        )
    payload = json.loads(ready_path.read_text(encoding="utf-8"))
    payload["checks"].pop("counter_direction")
    ready_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="required_check_keys"):
        trainer.validate_training_readiness(
            root=tmp_path,
            config_path=config_path,
            epochs=12,
            use_mock_dino=False,
            git_head="head-a",
            git_branch="test-branch",
            remote_head="head-a",
            clean_status="",
            source_tree_hash=python_source_tree_hash(tmp_path),
        )


def test_trainer_run_cannot_bypass_readiness_with_cli_flag() -> None:
    source = inspect.getsource(trainer.run)
    assert "if args.require_ready" not in source
    assert "validate_training_readiness(" in source


def test_meta_event_reports_cached_dino_calls_truthfully() -> None:
    source = inspect.getsource(trainer.run)
    assert "dino_calls=0" in source
    assert '"audit_field_cache_build_dino_calls"' in source


def test_pilot_history_survives_resume_and_replaces_same_epoch(tmp_path: Path) -> None:
    first = {"epoch": 0, "value": "old"}
    replacement = {"epoch": 0, "value": "new"}
    second = {"epoch": 1, "value": "second"}

    trainer._record_pilot_history(tmp_path, first)
    trainer._record_pilot_history(tmp_path, second)
    history = trainer._record_pilot_history(tmp_path, replacement)

    assert history == [replacement, second]
    assert trainer._load_pilot_history(tmp_path) == [replacement, second]


def test_trainer_saves_pre_eval_resume_checkpoint() -> None:
    source = inspect.getsource(trainer.run)
    pre_eval = source.index('"phase": "pre_eval"')
    audit = source.index("audit_outputs = collect_outputs", pre_eval)

    assert "micro_step=micro_batches_per_epoch" in source[pre_eval - 500 : audit]
