import torch

from fate_oia.models.aie_contribution_head import AIEContributionHead


def test_small_nonzero_head_initialization_activates_evidence_gradient():
    head = AIEContributionHead(dim=32)
    evidence = torch.randn(2, 4, 4, 32, requires_grad=True)
    out = head(evidence, torch.zeros(2, 4), action_scale=0.1)
    assert float(out["raw_contribution"].std()) > 0
    out["action_logits_final_train"].sum().backward()
    assert float(evidence.grad.abs().sum()) > 0

