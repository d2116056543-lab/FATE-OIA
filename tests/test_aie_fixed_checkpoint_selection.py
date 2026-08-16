import pytest
import torch

from fate_oia.engine.train_aie_oia import (
    checkpoint_selection_criteria,
    compatible_checkpoint_state_dict,
)


def test_checkpoint_selection_tracks_genuine_fixed_metrics_separately_from_deploy():
    selected = {
        "final": {"joint": 0.54, "Act_mF1": 0.72, "Exp_mF1": 0.36, "Act_mAP": 0.79, "Exp_mAP": 0.38},
        "deploy": {"joint": 0.57, "Act_mF1": 0.73, "Exp_mF1": 0.41},
    }

    criteria = checkpoint_selection_criteria(selected)

    assert criteria["fixed_joint"] == 0.54
    assert criteria["fixed_action_mF1"] == 0.72
    assert criteria["fixed_reason_mF1"] == 0.36
    assert criteria["deploy_joint"] == 0.57


def test_checkpoint_loader_drops_only_known_nonpersistent_grammar_buffers():
    model = torch.nn.Linear(2, 1)
    state = model.state_dict()
    state["foundation.predicate_reason.positive_mask"] = torch.ones(1)

    compatible = compatible_checkpoint_state_dict(model, state)

    assert set(compatible) == set(model.state_dict())

    state["unexpected.learned.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="unexpected learned state"):
        compatible_checkpoint_state_dict(model, state)
