import torch

from fate_oia.models.tida_reason_reader import TIDAReasonReader


def test_reason_delta_uses_attended_evidence_confidence_and_stays_bounded():
    torch.manual_seed(7)
    reader = TIDAReasonReader(dim=8, num_reasons=3, kappa=0.12, evidence_trust_cap=0.25)
    reason_nodes = torch.randn(2, 3, 8)
    predicate_state = torch.randn(2, 4, 8)
    action_innovation = torch.randn(2, 2, 8)
    reliability = torch.tensor(
        [[0.9, 0.6, 0.3, 0.1, 0.8, 0.2], [0.4, 0.7, 0.2, 0.5, 0.1, 0.9]]
    )

    output = reader(
        reason_nodes,
        predicate_state,
        action_innovation,
        reliability,
        temporal_scale=1.0,
    )
    route = output["reason_temporal_route"][..., :-1]
    expected_confidence = (route * reliability[:, None]).sum(-1)

    assert torch.allclose(output["reason_evidence_confidence"], expected_confidence)
    assert torch.allclose(
        output["reason_effective_trust"], 0.25 * expected_confidence
    )
    assert output["reason_temporal_delta"].abs().max() <= 0.25 * 0.12 + 1e-7


def test_lower_reliability_cannot_keep_the_same_reason_correction():
    torch.manual_seed(11)
    reader = TIDAReasonReader(dim=8, num_reasons=3, evidence_trust_cap=0.25)
    inputs = (
        torch.randn(2, 3, 8),
        torch.randn(2, 4, 8),
        torch.randn(2, 2, 8),
    )
    high = reader(*inputs, torch.full((2, 6), 0.9), temporal_scale=1.0)
    low = reader(*inputs, torch.full((2, 6), 0.1), temporal_scale=1.0)

    assert low["reason_evidence_confidence"].mean() < high["reason_evidence_confidence"].mean()
    assert low["reason_temporal_delta"].abs().mean() < high["reason_temporal_delta"].abs().mean()
