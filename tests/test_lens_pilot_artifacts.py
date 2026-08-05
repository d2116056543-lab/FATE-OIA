import json

import torch


def test_pilot_evaluator_recomputes_faithfulness_from_raw_tensor(tmp_path):
    from fate_oia.engine.evaluate_lens_oia_pilot import _faithfulness

    epoch = tmp_path / "epoch_00"
    epoch.mkdir()
    selection = torch.zeros(8, 4, 22)
    selection[..., 0] = 1
    contribution = torch.zeros(8, 4, 21)
    labels = torch.ones(8, 4)
    contribution[..., 0] = 0.5
    state = torch.zeros(8, 4, 21, 3)
    state[..., 0, 0] = 0.6
    state[..., 0, 1] = -0.2
    torch.save({"factor_selection": selection, "factor_contribution_bounded": contribution, "labels_action": labels, "factor_contribution_state": state}, epoch / "audit_subset.pt")
    result = _faithfulness(epoch / "audit_subset.pt")
    assert result["selected_deletion_effect"] > result["equal_mass_control_effect"]
    assert result["selected_direction_lcb95"] > 0
    assert result["state_swap_direction_lcb95"] > 0


def test_pilot_evaluator_does_not_self_declare_pass_without_artifacts(tmp_path):
    from fate_oia.engine.evaluate_lens_oia_pilot import evaluate

    result = evaluate(tmp_path)
    assert result["status"] == "PILOT_FAIL"
    assert not result["gates"]["F"]
    assert not result["gates"]["G"]

