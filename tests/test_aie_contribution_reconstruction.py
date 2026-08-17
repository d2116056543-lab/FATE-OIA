import torch
import pytest

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


def test_action_specific_scale_controls_each_action_without_breaking_reconstruction():
    torch.manual_seed(7)
    head = AIEContributionHead(dim=32, action_dim=4, probes_per_action=4, kappa=3.0)
    primary = torch.randn(3, 4)
    evidence = torch.randn(3, 4, 4, 32)

    full = head(evidence, primary, action_scale=1.0)
    scales = torch.tensor([0.0, 0.25, 0.75, 0.25])
    mixed = head(evidence, primary, action_scale=scales)

    torch.testing.assert_close(mixed["action_delta"][:, 0], torch.zeros(3), atol=1e-7, rtol=0)
    torch.testing.assert_close(
        mixed["action_logits_final"] - primary,
        mixed["bounded_contribution"].sum(-1),
        atol=1e-6,
        rtol=0,
    )
    assert float(mixed["contribution_reconstruction_error"]) < 1e-6
    assert not torch.equal(full["action_delta"][:, 1:], mixed["action_delta"][:, 1:])


def test_scalar_scale_remains_equivalent_to_uniform_action_vector():
    torch.manual_seed(11)
    head = AIEContributionHead(dim=16, action_dim=4, probes_per_action=2)
    primary = torch.randn(2, 4)
    evidence = torch.randn(2, 4, 2, 16)

    scalar = head(evidence, primary, action_scale=0.5)
    vector = head(evidence, primary, action_scale=torch.full((4,), 0.5))
    for key in ("action_logits_final", "action_logits_final_train", "bounded_contribution", "action_delta"):
        torch.testing.assert_close(scalar[key], vector[key], atol=0, rtol=0)


def test_action_specific_scale_rejects_wrong_length():
    head = AIEContributionHead(dim=16, action_dim=4, probes_per_action=2)
    with pytest.raises(ValueError, match="four action values"):
        head(torch.randn(2, 4, 2, 16), torch.randn(2, 4), action_scale=[0.5, 0.5])


