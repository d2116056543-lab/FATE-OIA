import inspect
from pathlib import Path

import yaml

from fate_oia.engine import train_acpr_meter_oia as trainer


def test_tesa_training_protocol_matches_fixed_plan():
    config = yaml.safe_load(
        Path(
            "configs/fate_oia_train_360x640_acpr_meter_oia_v2_tesa.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["training"]["epochs"] == 12
    assert config["training"]["batch_size"] == 6
    assert config["training"]["gradient_accumulation_steps"] == 5
    assert config["runtime"]["test_only"] is True
    assert config["runtime"]["one_dino_call"] is True
    assert config["runtime"]["sequential_eval"] is True
    assert config["posthoc_calibration"]["fit_split"] == "train_calib"
    assert config["best_selection_split"] == "test"
    assert config["meta"] == {"training_enabled": False, "audit_only": True}


def test_tesa_trainer_contains_dense_losses_and_no_v1_event_path():
    source = inspect.getsource(trainer._compute_losses)
    assert "dense_factor_intervention_loss" in source
    assert "identity_corruption_loss" in source
    assert "meter_grounding_loss" in source
    assert "meter_private_pu_loss" in source
    assert "_counterfactual_event" not in inspect.getsource(trainer)
