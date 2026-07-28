import torch

from fate_oia.models.meter_semantic_action import METERSemanticActionPeer


def test_semantic_logit_decomposition_is_exact() -> None:
    module = METERSemanticActionPeer(dim=16, action_dim=4, factor_dim=21)
    output = module(torch.randn(2, 4), torch.randn(2, 4, 16), torch.randn(2, 21, 16), torch.rand(2, 21), progress=1.0)
    reconstructed = output["semantic_bias"] + output["action_factor_contributions"].sum(-1)
    torch.testing.assert_close(output["action_logits_semantic"], reconstructed, atol=1e-6, rtol=0)
