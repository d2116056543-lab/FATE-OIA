from pathlib import Path

import yaml

import inspect

from fate_oia.engine import audit_acpr_meter_oia
from fate_oia.engine.audit_acpr_meter_oia import REQUIRED_FILES


def test_audit_contract_lists_the_formal_source_files() -> None:
    assert "fate_oia/engine/train_acpr_meter_oia.py" in REQUIRED_FILES
    assert "fate_oia/models/meter_oia_model.py" in REQUIRED_FILES
    assert "fate_oia/optim/meter_meta_utility.py" in REQUIRED_FILES


def test_formal_config_requires_matched_null_meta_admission() -> None:
    config = yaml.safe_load(
        Path(
            "configs/fate_oia_train_360x640_acpr_meter_oia_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["meta"]["admission_mode"] == "matched_null_lcb"
    assert config["meta"]["admission_audit_batches"] >= 2
    assert config["meta"]["admission_min_consecutive"] >= 2


def test_audit_requires_isolated_profile_and_all_event_memory_peaks() -> None:
    source = inspect.getsource(audit_acpr_meter_oia.run_audit)
    assert source.index("git_head = subprocess.check_output") < source.index(
        'profile.get("git_head") == git_head'
    )
    assert '"isolation_pass"' in source
    assert "child_exit_status" in source
    assert "memory_peak_after_ordinary_gb" in source
    assert "memory_peak_after_counterfactual_gb" in source
    assert "memory_peak_after_meta_gb" in source
    assert "memory_peak_after_calibration_gb" in source
    assert "recovery_verified" in source
    assert "profile_identity_ok" in source
    assert "and not clean_status" in source
    assert 'selected.get("meta_utility_finite")' in source
