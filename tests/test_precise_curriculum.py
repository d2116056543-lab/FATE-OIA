from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from fate_oia.engine.precise_curriculum import (
    curriculum_sha256,
    curriculum_state_for_epoch,
    owner_active_epoch_counts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "fate_oia_train_360x640_precise_oia_v1.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_fixed_twelve_epoch_curriculum_matches_approved_schedule() -> None:
    config = _config()
    expected = {
        0: ("foundation", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        1: ("foundation", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: ("grounding_ramp_1", 0.35, 0.35, 0.0, 0.0, 0.0, 0.50),
        3: ("grounding_ramp_2", 0.70, 0.70, 0.0, 0.0, 0.0, 1.00),
        4: ("joint_ramp_1", 1.0, 1.0, 0.35, 0.35, 0.25, 1.00),
        5: ("joint_ramp_2", 1.0, 1.0, 0.70, 0.70, 0.60, 1.00),
        6: ("safe_joint", 1.0, 1.0, 1.0, 1.0, 1.0, 1.00),
        11: ("safe_joint", 1.0, 1.0, 1.0, 1.0, 1.0, 1.00),
    }
    for epoch, values in expected.items():
        state = curriculum_state_for_epoch(config, epoch)
        assert (
            state.stage,
            state.reread,
            state.annotation,
            state.exchange,
            state.reason_latent,
            state.intervention,
            state.threshold,
        ) == values
        assert state.foundation == 1.0
        assert state.evidence == 1.0


def test_owner_activation_and_active_epoch_totals_are_local() -> None:
    config = _config()
    assert owner_active_epoch_counts(config, epochs=12) == {
        "action_foundation": 12,
        "action_decoder": 12,
        "reason_semantic": 12,
        "evidence_core": 12,
        "reread_adapter": 10,
        "annotation_adapter": 10,
        "threshold_head": 10,
        "exchange_adapter": 8,
        "reason_latent": 8,
    }
    assert not curriculum_state_for_epoch(config, 1).owner_active["reread_adapter"]
    assert curriculum_state_for_epoch(config, 2).owner_active["reread_adapter"]
    assert not curriculum_state_for_epoch(config, 3).owner_active["exchange_adapter"]
    assert curriculum_state_for_epoch(config, 4).owner_active["exchange_adapter"]
    assert all(curriculum_state_for_epoch(config, 6).owner_active.values())


def test_curriculum_hash_uses_normalized_content_not_mapping_order() -> None:
    config = _config()
    reordered = deepcopy(config)
    reordered["curriculum"] = dict(reversed(list(config["curriculum"].items())))
    assert curriculum_sha256(config) == curriculum_sha256(reordered)
    changed = deepcopy(config)
    changed["curriculum"]["schedule"][2]["reread"] = 0.36
    assert curriculum_sha256(config) != curriculum_sha256(changed)


@pytest.mark.parametrize("epoch", [-1, 12])
def test_curriculum_rejects_epoch_outside_formal_run(epoch: int) -> None:
    with pytest.raises(ValueError, match="epoch"):
        curriculum_state_for_epoch(_config(), epoch)
