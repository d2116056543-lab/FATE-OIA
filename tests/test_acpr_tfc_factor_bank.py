import copy

import pytest
import yaml

from fate_oia.models.tfc_factor_bank import TFCFactorBank


def test_tfc_factor_bank_contract():
    bank = TFCFactorBank.from_yaml("configs/acpr_tfc_factors.yaml")
    mats = bank.compatibility_matrices()
    assert bank.num_factors >= 10
    assert mats["factor_to_action_support"].shape == (bank.num_factors, 4)
    assert mats["factor_to_reason_support"].shape == (bank.num_factors, 21)
    assert mats["factor_conflict"].shape == (bank.num_factors, bank.num_factors)
    assert mats["native_similarity"].isfinite().all()
    assert bank.reason_alias_coverage.sum().item() == 21
    assert set(bank.reason_aliases.values()) == set(range(21))


def test_tfc_factor_bank_rejects_invalid_target_indices(tmp_path):
    with open("configs/acpr_tfc_factors.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bad = copy.deepcopy(cfg)
    first_factor = next(iter(bad["factors"].values()))
    first_factor["target_scope"]["reason_support"] = [99]
    path = tmp_path / "bad_factor.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="reason_support index out of range"):
        TFCFactorBank.from_yaml(path)


def test_tfc_factor_bank_rejects_action_target_schema_drift(tmp_path):
    with open("configs/acpr_tfc_factors.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bad = copy.deepcopy(cfg)
    bad["action_targets"] = ["stop_slow", "forward", "turn_left"]
    path = tmp_path / "bad_actions.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="action_targets must exactly define"):
        TFCFactorBank.from_yaml(path)


def test_tfc_factor_bank_rejects_incomplete_reason_alias_coverage(tmp_path):
    with open("configs/acpr_tfc_factors.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bad = copy.deepcopy(cfg)
    bad["reason_targets"]["aliases"].pop("bdd_oia_reason_20_unmapped")
    path = tmp_path / "bad_reason_aliases.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="reason_targets aliases must cover all"):
        TFCFactorBank.from_yaml(path)
