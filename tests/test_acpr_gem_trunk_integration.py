import torch

from fate_oia.models.acpr_label_trunk import ACPRLabelTrunk


def test_label_trunk_zero_init_evidence_is_equivalent_and_exposes_attention():
    trunk = ACPRLabelTrunk(dim=64, action_dim=4, reason_dim=21)
    patch = torch.randn(2, 3, 32, 64)
    evidence = torch.randn(2, 20, 64)

    base = trunk(patch)
    gem = trunk(patch, evidence_tokens=evidence, evidence_enabled=True)

    assert torch.allclose(base["action_logits_direct"], gem["action_logits_direct"], atol=1e-6)
    assert torch.allclose(base["reason_logits_visual"], gem["reason_logits_visual"], atol=1e-6)
    assert gem["label_evidence_attention"].shape == (2, 25, 20)
    assert gem["action_evidence_attention"].shape == (2, 4, 20)
    assert gem["reason_evidence_attention"].shape == (2, 21, 20)


def test_label_trunk_first_backward_reaches_evidence_output_projection():
    trunk = ACPRLabelTrunk(dim=64, action_dim=4, reason_dim=21)
    patch = torch.randn(2, 3, 32, 64)
    evidence = torch.randn(2, 20, 64)

    out = trunk(patch, evidence_tokens=evidence, evidence_enabled=True)
    loss = out["action_logits_direct"].sum() + out["reason_logits_visual"].sum()
    loss.backward()

    assert trunk.evidence_out_proj.weight.grad is not None
    assert trunk.evidence_out_proj.weight.grad.abs().sum() > 0
