import torch

from fate_oia.models.meter_semantic_action import METERSemanticActionPeer


def test_semantic_action_is_additive_and_final_warmup_is_exact() -> None:
    torch.manual_seed(8)
    module = METERSemanticActionPeer(dim=24, action_dim=4, factor_dim=21)
    visual = torch.randn(2, 4)
    action_nodes = torch.randn(2, 4, 24)
    factors = torch.randn(2, 21, 24)
    reliability = torch.rand(2, 21)

    zero = module(visual, action_nodes, factors, reliability, progress=0.0)
    full = module(visual, action_nodes, factors, reliability, progress=1.0)

    torch.testing.assert_close(zero["action_logits_final"], visual, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        full["action_logits_semantic"],
        module.semantic_bias.view(1, -1) + full["action_factor_contributions"].sum(-1),
        atol=1e-6,
        rtol=0,
    )
    assert full["action_factor_weights"].shape == (2, 4, 21)
    assert full["action_selector"].std() > 0
