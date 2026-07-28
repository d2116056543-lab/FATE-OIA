import torch

from fate_oia.models.meter_signed_factors import METERsignedFactors


def test_factor_reliability_is_bounded_and_null_is_exposed() -> None:
    module = METERsignedFactors(dim=16, factor_dim=21, num_layers=3, rank=4)
    output = module(torch.randn(2, 21, 16), torch.randn(2, 3, 12, 16), progress=1.0)
    assert torch.isfinite(output["factor_reliability"]).all()
    assert ((output["factor_reliability"] >= 0) & (output["factor_reliability"] <= 1)).all()
    assert output["factor_support_null"].shape == (2, 21)
    assert output["factor_counter_null"].shape == (2, 21)
