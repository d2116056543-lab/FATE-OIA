import torch

from fate_oia.models.acpr_label_trunk import ACPRLabelTrunk


def test_acpr_label_trunk_gate_and_shapes():
    trunk = ACPRLabelTrunk()
    out = trunk(torch.randn(2, 3, 3600, 384))
    assert out["action_logits_direct"].shape == (2, 4)
    assert out["reason_logits_visual"].shape == (2, 21)
    assert out["label_attention"].shape == (2, 25, 3600)
    assert float(out["action_fusion_gate"].min()) >= 0.10
    assert float(out["action_fusion_gate"].max()) <= 0.90
