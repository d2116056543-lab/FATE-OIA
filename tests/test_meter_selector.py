import torch

from fate_oia.models.meter_semantic_action import METERSemanticActionPeer


def test_selector_is_sample_and_action_dependent() -> None:
    torch.manual_seed(12)
    module = METERSemanticActionPeer(dim=16, action_dim=4, factor_dim=21)
    output = module(torch.randn(3, 4), torch.randn(3, 4, 16), torch.randn(3, 21, 16), torch.rand(3, 21), progress=1.0)
    assert output["action_selector"].shape == (3, 4)
    assert torch.isfinite(output["action_selector"]).all()
    assert output["action_selector"].std() > 0
