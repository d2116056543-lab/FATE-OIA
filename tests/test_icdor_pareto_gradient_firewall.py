from __future__ import annotations

import torch


def _grads(loss: torch.Tensor, *values: torch.Tensor) -> tuple[float, ...]:
    result = torch.autograd.grad(loss, values, allow_unused=True, retain_graph=True)
    return tuple(0.0 if grad is None else float(grad.abs().sum()) for grad in result)


def test_pareto_penalty_never_updates_base_and_has_primal_signal_at_zero_dual() -> None:
    from fate_oia.optim.mosaic_action_pareto_admission import MOSAICActionParetoAdmission

    base = torch.randn(3, 4, requires_grad=True)
    route_delta = torch.full((3, 4), 3.0, requires_grad=True)
    targets = torch.zeros(3, 4)
    routed = base.detach() + route_delta
    pareto = MOSAICActionParetoAdmission(tolerance=0.0)
    assert torch.count_nonzero(pareto.dual_variables) == 0

    primal = pareto.route_penalty(base, routed, targets)
    assert primal.item() > 0.0
    base_grad, route_grad = _grads(primal, base, route_delta)
    assert base_grad <= 1e-12
    assert route_grad >= 1e-8


def test_route_delta_construction_preserves_gradient_firewall() -> None:
    base = torch.randn(2, 4, requires_grad=True)
    shadow = base + torch.randn(2, 4, requires_grad=True)
    route_delta = shadow - base
    routed = base.detach() + route_delta
    routed.sum().backward()
    assert base.grad is None or base.grad.abs().max().item() <= 1e-12

