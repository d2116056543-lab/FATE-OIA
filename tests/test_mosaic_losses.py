from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.losses.mosaic_action_losses import (
    action_asymmetric_loss,
    action_cardinality_loss,
    build_mosaic_action_loss,
)
from fate_oia.losses.mosaic_factor_losses import build_mosaic_factor_loss
from fate_oia.losses.mosaic_reason_observation_losses import build_mosaic_reason_loss
from fate_oia.losses.mosaic_state_losses import build_mosaic_state_loss


def test_formal_mosaic_losses_do_not_use_autocast_unsafe_probability_bce() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1] / "fate_oia" / "losses"
    for path in root.glob("mosaic_*.py"):
        source = path.read_text(encoding="utf-8")
        assert "F.binary_cross_entropy(" not in source, path.name


def test_action_asl_is_the_true_clipped_asymmetric_formula_not_plain_bce() -> None:
    logits = torch.tensor([[2.0, -1.0]])
    targets = torch.tensor([[1.0, 0.0]])
    loss = action_asymmetric_loss(logits, targets, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    probability = torch.sigmoid(logits)
    negative_probability = (1.0 - probability + 0.05).clamp(max=1.0)
    manual = -(
        targets * torch.log(probability)
        + (1.0 - targets)
        * torch.log(negative_probability)
        * (1.0 - negative_probability).pow(4.0)
    ).mean()
    assert torch.allclose(loss, manual)
    assert not torch.allclose(loss, F.binary_cross_entropy_with_logits(logits, targets))


def test_action_loss_uses_declared_rank_and_cardinality_weights_once() -> None:
    logits = torch.randn(3, 4, requires_grad=True)
    targets = torch.randint(0, 2, (3, 4)).float()
    rank = logits.sum() * 0 + 0.7
    output = build_mosaic_action_loss(logits, targets, rank_loss=rank)
    expected = output["loss_action_asl"] + 0.10 * rank + 0.02 * action_cardinality_loss(logits, targets)
    assert torch.allclose(output["loss_action_total"], expected)


def test_action_loss_respects_explicit_runtime_weights() -> None:
    logits = torch.randn(3, 4)
    targets = torch.randint(0, 2, (3, 4)).float()
    rank = logits.sum() * 0 + 0.7
    output = build_mosaic_action_loss(
        logits,
        targets,
        rank_loss=rank,
        rank_weight=0.25,
        cardinality_weight=0.04,
    )
    expected = (
        output["loss_action_asl"]
        + 0.25 * output["loss_action_rank"]
        + 0.04 * output["loss_action_cardinality"]
    )
    assert torch.allclose(output["loss_action_total"], expected)


def test_factor_losses_respect_unknown_masks_and_emit_valid_counts() -> None:
    predictions = {
        "factor_presence_logits": torch.zeros(1, 2, requires_grad=True),
        "factor_visibility_logits": torch.zeros(1, 2, requires_grad=True),
        "factor_soft_masks": torch.full((1, 2, 45, 80), 0.5, requires_grad=True),
        "factor_presence_prob": torch.full((1, 2), 0.5),
        "factor_visibility_prob": torch.full((1, 2), 0.5),
        "prototype_weights": torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]),
        "prior_scale": torch.tensor([0.05, 0.05], requires_grad=True),
    }
    observations = {
        "presence_target": torch.tensor([[1.0, 0.0]]),
        "presence_mask": torch.tensor([[1.0, 0.0]]),
        "visibility_target": torch.tensor([[1.0, 0.0]]),
        "visibility_mask": torch.tensor([[1.0, 0.0]]),
        "source_reliability": torch.tensor([[0.8, 0.0]]),
        "geometry_mask": torch.zeros(1, 2, 45, 80),
        "geometry_mask_valid": torch.tensor([[0.0, 0.0]]),
    }
    output = build_mosaic_factor_loss(predictions, observations)
    assert output["count_presence"] == 1
    assert output["count_visibility"] == 1
    assert output["count_geometry_mask"] == 0
    assert output["loss_geometry_mask"] == 0
    output["loss_factor_total"].backward()
    assert predictions["factor_presence_logits"].grad[0, 1] == 0


def test_reason_observation_losses_use_live_q_only_for_hidden_recovery_and_detached_q_for_m_step() -> None:
    logits = torch.zeros(2, 3, requires_grad=True)
    observed = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    observation_probability = torch.full((2, 3), 0.4, requires_grad=True)
    posterior_live = torch.full((2, 3), 0.6, requires_grad=True)
    posterior_detached = posterior_live.detach()
    hidden_mask = torch.tensor([[True, False, False], [False, False, False]])
    propensity = torch.full((2, 3), 0.5, requires_grad=True)
    output = build_mosaic_reason_loss(
        logits,
        observed,
        observation_probability,
        posterior_detached,
        posterior_live,
        propensity,
        hidden_mask,
        propensity_visibility_slopes=torch.ones(4, requires_grad=True),
        propensity_uncertainty_slopes=torch.ones(4, requires_grad=True),
        propensity_pi_min=0.20,
        propensity_pi_max=0.95,
        reason_false_positive_rate=torch.full((3,), 0.01, requires_grad=True),
        reason_false_positive_max=torch.full((3,), 0.05),
        rank_loss=logits.sum() * 0 + 0.2,
    )
    assert output["count_missing_recovery"] == 1
    assert torch.allclose(output["loss_missing_recovery"], -torch.log(torch.tensor(0.6)))
    output["loss_reason_total"].backward()
    assert posterior_live.grad is not None and posterior_live.grad[0, 0] < 0


def test_reason_loss_uses_squared_prevalence_hinges_and_exact_propensity_regularizer() -> None:
    logits = torch.full((2, 3), 4.0, requires_grad=True)
    observed = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    propensity = torch.tensor([[0.21, 0.50, 0.94], [0.21, 0.50, 0.94]], requires_grad=True)
    visibility_slopes = torch.tensor([1.0, 2.0], requires_grad=True)
    uncertainty_slopes = torch.tensor([2.0, 1.0], requires_grad=True)
    epsilon = torch.tensor([0.01, 0.0, 0.02], requires_grad=True)
    epsilon_max = torch.tensor([0.05, 0.0, 0.05])
    output = build_mosaic_reason_loss(
        logits,
        observed,
        torch.full((2, 3), 0.4),
        torch.full((2, 3), 0.5),
        torch.full((2, 3), 0.5),
        propensity,
        torch.zeros_like(observed, dtype=torch.bool),
        propensity_visibility_slopes=visibility_slopes,
        propensity_uncertainty_slopes=uncertainty_slopes,
        propensity_pi_min=0.20,
        propensity_pi_max=0.95,
        reason_false_positive_rate=epsilon,
        reason_false_positive_max=epsilon_max,
    )
    expected_slope = (visibility_slopes.square() + uncertainty_slopes.square()).mean()
    expected_boundary = (
        F.relu(0.05 - (propensity - 0.20)).square()
        + F.relu(0.05 - (0.95 - propensity)).square()
    ).mean()
    expected_epsilon = torch.tensor([(0.01 / 0.05) ** 2, 0.0, (0.02 / 0.05) ** 2]).mean()
    expected = expected_slope + 0.10 * expected_boundary + 0.10 * expected_epsilon
    assert torch.allclose(output["loss_propensity_slope"], expected_slope)
    assert torch.allclose(output["loss_propensity_boundary"], expected_boundary)
    assert torch.allclose(output["loss_propensity_epsilon"], expected_epsilon)
    assert torch.allclose(output["loss_propensity_regularization"], expected)
    output["loss_reason_total"].backward()
    assert visibility_slopes.grad is not None and visibility_slopes.grad.abs().sum() > 0
    assert uncertainty_slopes.grad is not None and uncertainty_slopes.grad.abs().sum() > 0
    assert epsilon.grad is not None and epsilon.grad.abs().sum() > 0


def test_state_loss_has_only_sparsity_residual_and_uncertainty_terms() -> None:
    output = build_mosaic_state_loss(
        {
            "decision_state_prob": torch.full((2, 8), 0.3, requires_grad=True),
            "decision_state_residual": torch.full((2, 8), 0.1, requires_grad=True),
            "decision_state_uncertainty": torch.full((2, 8), 0.2, requires_grad=True),
        }
    )
    expected = (
        0.02 * output["loss_state_sparsity"]
        + 0.02 * output["loss_state_residual"]
        + 0.02 * output["loss_state_uncertainty"]
    )
    assert torch.allclose(output["loss_state_total"], expected)


def test_state_loss_respects_explicit_runtime_weights() -> None:
    output = build_mosaic_state_loss(
        {
            "decision_state_prob": torch.full((2, 8), 0.3),
            "decision_state_residual": torch.full((2, 8), 0.1),
            "decision_state_uncertainty": torch.full((2, 8), 0.2),
        },
        sparsity_weight=0.03,
        residual_weight=0.04,
        uncertainty_weight=0.05,
    )
    expected = (
        0.03 * output["loss_state_sparsity"]
        + 0.04 * output["loss_state_residual"]
        + 0.05 * output["loss_state_uncertainty"]
    )
    assert torch.allclose(output["loss_state_total"], expected)


def test_factor_contradiction_loss_uses_declared_cross_factor_edges() -> None:
    base_predictions = {
        "factor_presence_logits": torch.zeros(1, 2),
        "factor_visibility_logits": torch.zeros(1, 2),
        "factor_soft_masks": torch.full((1, 2, 45, 80), 0.5),
        "factor_presence_prob": torch.full((1, 2), 0.5),
        "factor_visibility_prob": torch.full((1, 2), 0.5),
        "prototype_weights": torch.full((1, 2, 2), 0.5),
        "prior_scale": torch.zeros(2),
        "factor_contradiction_mask": torch.tensor([[False, True], [True, False]]),
    }
    observations = {
        "presence_target": torch.zeros(1, 2), "presence_mask": torch.zeros(1, 2),
        "visibility_target": torch.zeros(1, 2), "visibility_mask": torch.zeros(1, 2),
        "source_reliability": torch.zeros(1, 2),
        "geometry_mask": torch.zeros(1, 2, 45, 80), "geometry_mask_valid": torch.zeros(1, 2),
    }
    low = build_mosaic_factor_loss(
        {**base_predictions, "factor_positive_evidence": torch.tensor([[0.9, 0.0]])}, observations
    )
    high = build_mosaic_factor_loss(
        {**base_predictions, "factor_positive_evidence": torch.tensor([[0.9, 0.8]])}, observations
    )
    assert low["loss_contradiction"] == 0
    assert torch.allclose(high["loss_contradiction"], torch.tensor(0.72))
    assert high["count_contradiction"] == 1
