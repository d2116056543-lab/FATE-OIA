import torch

from fate_oia.models.acpr_label_trunk import ACPRLabelTrunk


def test_action_visual_head_is_per_action_token():
    trunk = ACPRLabelTrunk(dim=16)
    action_nodes = torch.randn(1, 4, 16)
    with torch.no_grad():
        base = trunk.action_visual_head(action_nodes).squeeze(-1)
        changed = action_nodes.clone()
        changed[:, 0, 0] += 10.0
        perturbed = trunk.action_visual_head(changed).squeeze(-1)
    delta = (perturbed - base).abs().squeeze(0)
    assert delta[0] == delta.max()
    assert delta[0] > 1e-4


def test_action_visual_logits_shape_from_forward():
    trunk = ACPRLabelTrunk()
    out = trunk(torch.randn(2, 3, 3600, 384))
    assert out["action_visual_logits"].shape == (2, 4)
    assert out["action_logits_direct"].shape == (2, 4)
