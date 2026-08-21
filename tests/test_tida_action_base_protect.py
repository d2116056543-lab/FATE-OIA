import torch

from fate_oia.losses.tida_losses import action_base_protect_loss


def test_action_base_protect_penalizes_gt_margin_regression_beyond_trust_region():
    image_logits = torch.tensor([[1.0, -1.0]])
    video_logits = torch.tensor([[0.99, -0.99]])
    targets = torch.tensor([[1.0, 0.0]])
    reliability = torch.zeros(1, 3)

    loss = action_base_protect_loss(image_logits, video_logits, targets, reliability)

    assert loss.item() > 0.0


def test_action_base_protect_allows_only_a_small_trust_region():
    image_logits = torch.tensor([[1.0, -1.0]])
    video_logits = torch.tensor([[0.997, -0.997]])
    targets = torch.tensor([[1.0, 0.0]])
    reliability = torch.zeros(1, 3)

    loss = action_base_protect_loss(image_logits, video_logits, targets, reliability)

    assert loss.item() == 0.0
