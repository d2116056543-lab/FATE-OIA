from __future__ import annotations

import torch

from fate_oia.losses.mosaic_icdor_action_losses import action_base_losses, action_route_losses
from fate_oia.optim.mosaic_action_pareto_admission import MOSAICActionParetoAdmission


def test_icdor_action_route_loss_detaches_visual_base_and_uses_true_asl() -> None:
    visual = torch.randn(4, 4, requires_grad=True)
    support = torch.rand(4, 4, requires_grad=True)
    veto = torch.rand(4, 4, requires_grad=True)
    targets = torch.randint(0, 2, (4, 4), dtype=torch.float32)
    base = action_base_losses(visual, targets)
    pareto = MOSAICActionParetoAdmission(action_count=4)
    route = action_route_losses(
        visual,
        support,
        veto,
        targets,
        support_dustbin=torch.rand(4, 4),
        veto_dustbin=torch.rand(4, 4),
        pareto_penalty=pareto.route_penalty(
            visual,
            visual.detach() + support - veto,
            targets,
        ),
        matched_random_logits=visual.detach() + support.detach().roll(1, 0) - veto.detach().roll(1, 0),
    )
    assert torch.isfinite(base["loss_action_base_total"])
    assert torch.isfinite(route["loss_action_route_total"])
    assert torch.isfinite(route["loss_action_route_intervention"])
    route["loss_action_route_total"].backward()
    assert visual.grad is None
    assert support.grad is not None and torch.count_nonzero(support.grad) > 0
    assert veto.grad is not None and torch.count_nonzero(veto.grad) > 0


def test_icdor_pareto_penalty_is_not_a_detached_noop() -> None:
    pareto = MOSAICActionParetoAdmission(action_count=4)
    pareto.dual_variables.fill_(1.0)
    visual = torch.tensor([[2.0, -2.0, 2.0, -2.0]], requires_grad=True)
    routed = torch.tensor([[-2.0, 2.0, -2.0, 2.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    penalty = pareto.route_penalty(visual, routed, targets)
    penalty.backward()
    assert penalty.item() > 0.0
    assert visual.grad is None
    assert routed.grad is not None and torch.count_nonzero(routed.grad) > 0


def test_icdor_pareto_duals_update_per_action_and_roundtrip_state() -> None:
    pareto = MOSAICActionParetoAdmission(action_count=4, tolerance=0.001, dual_lr=0.05)
    visual_ap = torch.tensor([0.70, 0.71, 0.72, 0.73])
    routed_ap = torch.tensor([0.68, 0.711, 0.719, 0.74])
    stats = pareto.update_from_audit(visual_ap, routed_ap)
    assert stats["pareto_violation_rate"] == 0.5
    assert pareto.dual_variables[0] > 0
    assert pareto.dual_variables[1] == 0
    restored = MOSAICActionParetoAdmission(action_count=4)
    restored.load_state_dict(pareto.state_dict())
    assert torch.equal(restored.dual_variables, pareto.dual_variables)
