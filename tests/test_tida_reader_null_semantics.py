import torch

from fate_oia.models.tida_action_reader import TIDAActionReader
from fate_oia.models.tida_reason_reader import TIDAReasonReader


def test_action_null_competes_with_complementary_reliability() -> None:
    reader = TIDAActionReader(dim=8)
    with torch.no_grad():
        reader.action_query.weight.zero_()
        reader.action_query.bias.zero_()
        reader.factor_key.weight.zero_()
        reader.factor_key.bias.zero_()
        reader.null_key.zero_()
    output = reader(
        torch.randn(2, 4, 8),
        torch.randn(2, 32, 8),
        torch.randn(2, 4, 8),
        torch.full((2, 36), 0.30),
        temporal_scale=1.0,
    )

    assert torch.all(output["action_nonnull_mass"] > 0.60)
    assert torch.all(output["action_nonnull_mass"] < 1.0)


def test_reason_reader_has_explicit_null_and_exact_zero_fallback() -> None:
    reader = TIDAReasonReader(dim=8)
    output = reader(
        torch.randn(2, 21, 8),
        torch.randn(2, 32, 8),
        torch.randn(2, 4, 8),
        torch.zeros(2, 36),
        temporal_scale=1.0,
    )

    assert output["reason_temporal_route"].shape == (2, 21, 37)
    assert torch.equal(output["reason_temporal_route"][..., -1], torch.ones(2, 21))
    assert torch.equal(output["reason_temporal_delta"], torch.zeros(2, 21))
    assert torch.equal(output["reason_temporal_evidence"], torch.zeros(2, 21, 8))


def test_reason_route_is_normalized_with_nonzero_evidence() -> None:
    reader = TIDAReasonReader(dim=8)
    output = reader(
        torch.randn(2, 21, 8),
        torch.randn(2, 32, 8),
        torch.randn(2, 4, 8),
        torch.full((2, 36), 0.30),
        temporal_scale=1.0,
    )

    assert torch.allclose(output["reason_temporal_route"].sum(-1), torch.ones(2, 21), atol=1e-6)
    assert output["reason_temporal_evidence"].shape == (2, 21, 8)
