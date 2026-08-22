import torch

from fate_oia.engine.evaluate_tida_oia import gt_margin_advantage


def test_gt_margin_advantage_is_positive_when_real_history_improves_target_margin():
    real = torch.tensor([[2.0, -2.0]])
    changed = torch.tensor([[1.0, -1.0]])
    target = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(gt_margin_advantage(real, changed, target), torch.ones(1, 2))


def test_gt_margin_advantage_exposes_harmful_history_by_label():
    real = torch.tensor([[0.0, 0.0]])
    changed = torch.tensor([[1.0, -1.0]])
    target = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(gt_margin_advantage(real, changed, target), -torch.ones(1, 2))
