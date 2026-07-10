from __future__ import annotations

import inspect

import pytest
import torch

from fate_oia.optim.mosaic_action_anchor import MOSAICActionAnchoredGradient


def test_conflicting_aux_gradient_is_scaled_to_the_action_halfspace_boundary() -> None:
    shared = torch.nn.Parameter(torch.tensor(0.0))
    helper = MOSAICActionAnchoredGradient(aux_shared_lambda_max=0.25, action_anchor_kappa=0.70)
    action_loss = 0.5 * (shared - 1.0).square()
    explanation_loss = 5.0 * (shared + 1.0).square()

    stats = helper.backward(action_loss, explanation_loss, [shared], [], [], step=7)

    assert stats["lambda_star"] == pytest.approx(0.03, rel=1e-5)
    assert stats["constraint_pass"] is True
    assert stats["halfspace_lhs"] >= stats["halfspace_rhs"] - 1e-6
    assert shared.grad.item() == pytest.approx(-0.70, abs=1e-5)
    assert stats["step"] == 7
    assert stats["shared_param_count"] == 1


def test_cooperative_gradient_uses_the_configured_maximum_aux_weight() -> None:
    shared = torch.nn.Parameter(torch.tensor(0.0))
    helper = MOSAICActionAnchoredGradient()
    action_loss = 0.5 * (shared - 1.0).square()
    explanation_loss = 0.5 * (shared - 2.0).square()
    stats = helper.backward(action_loss, explanation_loss, [shared], [], [], step=0)
    assert stats["dot_action_aux"] > 0
    assert stats["lambda_star"] == pytest.approx(0.25)
    assert shared.grad.item() == pytest.approx(-1.5)


def test_task_specific_parameters_receive_only_their_own_losses() -> None:
    shared = torch.nn.Parameter(torch.tensor(0.5))
    action_only = torch.nn.Parameter(torch.tensor(0.5))
    explanation_only = torch.nn.Parameter(torch.tensor(0.5))
    helper = MOSAICActionAnchoredGradient()
    action_loss = (shared + action_only).square()
    explanation_loss = (shared + explanation_only - 2.0).square()
    helper.backward(action_loss, explanation_loss, [shared], [action_only], [explanation_only], step=0)
    assert action_only.grad.item() == pytest.approx(2.0)
    assert explanation_only.grad.item() == pytest.approx(-2.0)


def test_gradient_accumulation_adds_without_overwriting_prior_microbatch() -> None:
    shared = torch.nn.Parameter(torch.tensor(0.0))
    action_only = torch.nn.Parameter(torch.tensor(0.0))
    helper = MOSAICActionAnchoredGradient()
    for _ in range(2):
        action_loss = (shared + action_only - 1.0).square()
        explanation_loss = (shared + 1.0).square()
        helper.accumulate(
            action_loss,
            explanation_loss,
            [shared],
            [action_only],
            [],
            loss_scale=0.5,
        )
    stats = helper.finalize(step=0)
    assert shared.grad is not None
    first_total = shared.grad.item()
    assert first_total != 0
    assert action_only.grad.item() == pytest.approx(-2.0)
    assert stats["microbatch_count"] == 2


def test_parameter_partitions_must_be_unique_disjoint_and_trainable() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    helper = MOSAICActionAnchoredGradient()
    with pytest.raises(ValueError, match="disjoint"):
        helper.backward(parameter.square(), parameter.square(), [parameter], [parameter], [], step=0)
    frozen = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
    with pytest.raises(ValueError, match="trainable"):
        helper.backward(parameter.square(), parameter.square(), [frozen], [], [], step=0)


def test_accumulation_projects_the_aggregate_window_halfspace() -> None:
    shared = torch.nn.Parameter(torch.zeros(2))
    action_only = torch.nn.Parameter(torch.zeros(1))
    explanation_only = torch.nn.Parameter(torch.zeros(1))
    helper = MOSAICActionAnchoredGradient(aux_shared_lambda_max=0.25, action_anchor_kappa=0.70)
    for action_vector, explanation_vector in zip(
        (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
        (torch.tensor([-10.0, 0.0]), torch.tensor([0.0, -10.0])),
    ):
        helper.accumulate(
            (shared * action_vector).sum() + action_only.sum() * 0,
            (shared * explanation_vector).sum() + explanation_only.sum() * 0,
            [shared], [action_only], [explanation_only], loss_scale=1.0,
        )
    stats = helper.finalize(step=0)
    aggregate_action = torch.tensor([1.0, 1.0])
    assert torch.dot(shared.grad, aggregate_action) + 1e-6 >= 0.70 * aggregate_action.square().sum()
    assert stats["constraint_pass"]
    assert stats["microbatch_count"] == 2
    assert ".item()" not in inspect.getsource(MOSAICActionAnchoredGradient.accumulate)
