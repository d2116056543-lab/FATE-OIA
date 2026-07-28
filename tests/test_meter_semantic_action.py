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


def test_semantic_action_cannot_collapse_into_null_only_bias() -> None:
    torch.manual_seed(19)
    module = METERSemanticActionPeer(dim=24, action_dim=4, factor_dim=21)
    output = module(
        torch.randn(8, 4),
        torch.randn(8, 4, 24),
        torch.randn(8, 21, 24),
        torch.full((8, 21), 0.5),
        progress=1.0,
    )

    assert 0.01 < float(output["action_null_mass"].mean()) < 0.80
    assert float(output["action_factor_weights"].sum(-1).mean()) > 0.20
    assert float(output["action_factor_contributions"].square().mean().sqrt()) > 1e-4
    assert float(output["action_logits_semantic"].std(dim=0).mean()) > 1e-4


def test_complete_semantic_probe_is_preserved_but_transport_is_bounded() -> None:
    torch.manual_seed(29)
    module = METERSemanticActionPeer(dim=24, action_dim=4, factor_dim=21)
    visual = torch.randn(16, 4)
    with torch.no_grad():
        module.factor_value.mul_(50.0)
    output = module(
        visual,
        torch.randn(16, 4, 24),
        torch.randn(16, 21, 24),
        torch.ones(16, 21),
        progress=1.0,
        update_running_stats=True,
    )

    torch.testing.assert_close(
        output["action_logits_semantic"],
        output["semantic_bias"] + output["action_factor_contributions"].sum(-1),
        atol=1e-6,
        rtol=0,
    )
    ratio = (
        output["action_semantic_transport_delta"].square().mean(dim=0).sqrt()
        / visual.square().mean(dim=0).sqrt().clamp_min(1e-6)
    )
    assert bool(((ratio >= 0.03) & (ratio <= 0.30)).all())
    assert output["semantic_transport_actual_ratio"].shape == (4,)
    torch.testing.assert_close(
        output["action_logits_semantic_transport"],
        visual + output["action_semantic_transport_delta"],
    )


def test_semantic_transport_is_batch_composition_invariant_in_eval() -> None:
    torch.manual_seed(31)
    module = METERSemanticActionPeer(dim=24, action_dim=4, factor_dim=21)
    module.running_visual_rms.copy_(torch.tensor([1.5, 1.0, 2.0, 0.8]))
    module.running_semantic_delta_rms.copy_(torch.tensor([4.0, 2.0, 3.0, 5.0]))
    module.running_rms_updates.fill_(3)
    module.eval()
    visual = torch.randn(1, 4)
    action_nodes = torch.randn(1, 4, 24)
    factors = torch.randn(1, 21, 24)
    reliability = torch.rand(1, 21)

    single = module(visual, action_nodes, factors, reliability, progress=1.0)
    mixed = module(
        torch.cat([visual, torch.randn(7, 4)], dim=0),
        torch.cat([action_nodes, torch.randn(7, 4, 24)], dim=0),
        torch.cat([factors, torch.randn(7, 21, 24)], dim=0),
        torch.cat([reliability, torch.rand(7, 21)], dim=0),
        progress=1.0,
    )

    torch.testing.assert_close(
        single["action_logits_semantic_transport"],
        mixed["action_logits_semantic_transport"][:1],
    )
