import torch

from fate_oia.models.vetra_tta_combo_calibrator import VetraTTAComboCalibrator, remap_horizontal_flip_actions


def test_flip_remap_swaps_only_left_and_right():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(remap_horizontal_flip_actions(logits), torch.tensor([[1.0, 2.0, 4.0, 3.0]]))


def test_combo_marginal_and_threshold_deploy_shapes():
    model = VetraTTAComboCalibrator(
        mean=torch.zeros(4), scale=torch.ones(4), coefficient=torch.zeros(3, 4),
        intercept=torch.tensor([0.0, 1.0, 0.0]), class_codes=torch.tensor([1, 5, 9]),
        thresholds=torch.full((4,), 0.5), original_weight=0.75,
    )
    output = model(torch.zeros(2, 4), torch.zeros(2, 4))
    assert output["action_deploy_logits"].shape == (2, 4)
    assert torch.allclose(output["combo_probs"].sum(-1), torch.ones(2))
    assert torch.isfinite(output["action_deploy_logits"]).all()
