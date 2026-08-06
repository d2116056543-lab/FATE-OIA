import torch

from fate_oia.models.aie_contribution_head import AIEContributionHead


def test_train_final_detaches_primary_but_inference_keeps_value():
    head = AIEContributionHead(dim=16)
    primary = torch.randn(1, 4, requires_grad=True)
    out = head(torch.randn(1, 4, 4, 16, requires_grad=True), primary)
    torch.testing.assert_close(out["action_logits_final"], out["action_logits_final_train"])
    out["action_logits_final_train"].sum().backward()
    assert primary.grad is None

