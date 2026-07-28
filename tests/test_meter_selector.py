import torch

from fate_oia.losses.meter_action_losses import meter_action_loss
from fate_oia.models.meter_semantic_action import METERSemanticActionPeer


def test_selector_is_sample_and_action_dependent() -> None:
    torch.manual_seed(12)
    module = METERSemanticActionPeer(dim=16, action_dim=4, factor_dim=21)
    output = module(torch.randn(3, 4), torch.randn(3, 4, 16), torch.randn(3, 21, 16), torch.rand(3, 21), progress=1.0)
    assert output["action_selector"].shape == (3, 4)
    assert torch.isfinite(output["action_selector"]).all()
    assert output["action_selector"].std() > 0


def test_selector_regret_updates_selector_only() -> None:
    torch.manual_seed(41)
    module = METERSemanticActionPeer(dim=16, action_dim=4, factor_dim=21)
    visual = torch.randn(8, 4, requires_grad=True)
    action_nodes = torch.randn(8, 4, 16, requires_grad=True)
    factors = torch.randn(8, 21, 16, requires_grad=True)
    output = module(
        visual,
        action_nodes,
        factors,
        torch.rand(8, 21),
        progress=1.0,
    )
    loss = meter_action_loss(output, torch.randint(0, 2, (8, 4)).float())[
        "selector_regret"
    ]
    loss.backward()

    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in module.selector.parameters()
    )
    assert visual.grad is None or torch.equal(visual.grad, torch.zeros_like(visual))
    assert action_nodes.grad is None or torch.equal(
        action_nodes.grad, torch.zeros_like(action_nodes)
    )
    assert factors.grad is None or torch.equal(factors.grad, torch.zeros_like(factors))
