import inspect
from pathlib import Path

import yaml

from fate_oia.engine import train_acpr_meter_oia as trainer


def test_heca_training_protocol_matches_fixed_plan():
    config = yaml.safe_load(
        Path(
            "configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["training"]["epochs"] == 14
    assert config["training"]["batch_size"] == 4
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["training"]["shared_gradient_policy"] == "next_window_single_backward"
    assert config["runtime"]["test_only"] is True
    assert config["runtime"]["one_dino_call"] is True
    assert config["runtime"]["sequential_eval"] is False
    assert config["best_selection_split"] == "test"
    assert config["model"]["use_hard_action_factor_mask"] is False
    assert config["model"]["use_admission_gate"] is False


def test_heca_trainer_uses_private_reason_and_no_legacy_transport_path():
    source = inspect.getsource(trainer._compute_losses)
    assert "meter_action_loss" in source
    assert "meter_reason_loss" in source
    assert "meter_private_pu_loss" in source
    assert "observability=output[\"factor_observability\"].detach()" in source
    assert "FactorSpecificActionTransport" not in inspect.getsource(trainer)
