import torch

from fate_oia.models.aie_contribution_head import AIEContributionHead


def test_contributions_exactly_reconstruct_action_delta():
    head = AIEContributionHead(dim=32, action_dim=4, probes_per_action=4, kappa=3.0)
    primary = torch.randn(3, 4, requires_grad=True)
    evidence = torch.randn(3, 4, 4, 32, requires_grad=True)
    out = head(evidence, primary, action_scale=0.7)
    torch.testing.assert_close(
        out["action_logits_final"] - primary,
        out["bounded_contribution"].sum(-1),
        atol=1e-6,
        rtol=0,
    )
    assert float(out["contribution_reconstruction_error"]) < 1e-6
    out["action_logits_final_train"].sum().backward()
    assert primary.grad is None
    assert evidence.grad is not None and float(evidence.grad.abs().sum()) > 0


