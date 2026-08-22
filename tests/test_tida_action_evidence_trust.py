import torch

from fate_oia.models.tida_action_reader import TIDAActionReader


def test_action_delta_uses_attended_evidence_confidence_and_exact_contributions():
    torch.manual_seed(17)
    reader = TIDAActionReader(
        dim=8,
        num_actions=4,
        num_predicates=5,
        evidence_trust_cap=0.25,
    )
    with torch.no_grad():
        reader.action_output_weight.normal_()
    output = reader(
        torch.randn(2, 4, 8),
        torch.randn(2, 5, 8),
        torch.randn(2, 4, 8),
        torch.rand(2, 9),
        temporal_scale=1.0,
    )
    expected = (
        output["action_route"][..., :-1]
        * output["action_factor_reliability"][:, None, :-1]
    ).sum(-1)

    assert torch.allclose(output["action_evidence_confidence"], expected)
    assert torch.allclose(output["action_effective_trust"], 0.25 * expected)
    assert torch.allclose(
        output["action_factor_contribution"].sum(-1),
        output["action_temporal_delta"],
        atol=1e-6,
    )
    assert output["action_temporal_delta"].abs().max() <= 0.25 * 0.15 + 1e-7


def test_action_zero_reliability_keeps_exact_zero_delta():
    torch.manual_seed(19)
    reader = TIDAActionReader(dim=8, num_predicates=5, evidence_trust_cap=0.25)
    with torch.no_grad():
        reader.action_output_weight.normal_()
    output = reader(
        torch.randn(2, 4, 8),
        torch.randn(2, 5, 8),
        torch.randn(2, 4, 8),
        torch.zeros(2, 9),
        temporal_scale=1.0,
    )

    assert torch.equal(output["action_evidence_confidence"], torch.zeros(2, 4))
    assert torch.equal(output["action_temporal_delta"], torch.zeros(2, 4))
