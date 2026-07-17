import torch

from fate_oia.models.mosaic_continuous_credibility import ContinuousVisualCredibility


def test_continuous_credibility_is_available_without_certificate():
    module = ContinuousVisualCredibility(factor_count=3, dim=8)
    output = module(torch.randn(2, 3, 8), torch.rand(2, 3), torch.rand(2, 3))
    assert output["cV"].shape == (2, 3)
    assert torch.isfinite(output["cV"]).all()
    assert output["cV"].max() > 0

