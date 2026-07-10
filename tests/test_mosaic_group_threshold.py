from __future__ import annotations

import inspect

import torch

from fate_oia.models.mosaic_group_threshold import MOSAICGroupThresholdHead


TAIL_REASONS = [12, 9, 5, 14, 6, 11, 10, 13]


def test_group_threshold_uses_action_common_reason_and_tail_reason_groups() -> None:
    head = MOSAICGroupThresholdHead(tail_reason_indices=TAIL_REASONS)
    assert head.label_group_ids.shape == (25,)
    assert torch.equal(head.label_group_ids[:4], torch.zeros(4, dtype=torch.long))
    for reason_id in range(21):
        expected = 2 if reason_id in TAIL_REASONS else 1
        assert head.label_group_ids[4 + reason_id].item() == expected
    assert head.theta_group.shape == (3,)
    assert head.theta_delta.shape == (25,)


def test_deploy_equation_is_exact_raw_minus_fixed_theta() -> None:
    head = MOSAICGroupThresholdHead(tail_reason_indices=TAIL_REASONS)
    with torch.no_grad():
        head.theta_group.copy_(torch.tensor([0.1, -0.2, -0.4]))
        head.theta_delta.copy_(torch.linspace(-0.5, 0.5, 25))
    action = torch.randn(3, 4)
    reason = torch.randn(3, 21)
    output = head(action, reason)
    raw = torch.cat((action, reason), dim=-1)
    assert torch.equal(output["logits_raw"], raw)
    assert torch.allclose(output["logits_deploy"], raw - output["threshold_logit"].unsqueeze(0))
    assert torch.equal(output["action_logits_deploy"], output["logits_deploy"][:, :4])
    assert torch.equal(output["reason_logits_deploy"], output["logits_deploy"][:, 4:])


def test_calibration_backward_updates_only_group_and_label_thresholds() -> None:
    head = MOSAICGroupThresholdHead(tail_reason_indices=TAIL_REASONS)
    action = torch.randn(2, 4, requires_grad=True)
    reason = torch.randn(2, 21, requires_grad=True)
    output = head(action, reason)
    output["logits_deploy"].square().mean().backward()
    assert action.grad is None
    assert reason.grad is None
    assert head.theta_group.grad is not None and head.theta_group.grad.abs().sum() > 0
    assert head.theta_delta.grad is not None and head.theta_delta.grad.abs().sum() > 0


def test_thresholds_are_batch_independent_group_shrunk_and_bounded() -> None:
    head = MOSAICGroupThresholdHead(tail_reason_indices=TAIL_REASONS, label_delta_max=1.0)
    with torch.no_grad():
        head.theta_group.fill_(100.0)
        head.theta_delta.fill_(100.0)
    one = head(torch.zeros(1, 4), torch.zeros(1, 21))
    many = head(torch.randn(8, 4), torch.randn(8, 21))
    assert torch.equal(one["threshold_logit"], many["threshold_logit"])
    assert torch.all(one["threshold_logit"] <= head.threshold_max_logit)
    assert torch.all(one["threshold_logit"] >= head.threshold_min_logit)
    assert torch.all(head.label_delta.abs() <= 1.0)


def test_calibration_objective_matches_the_five_declared_terms() -> None:
    head = MOSAICGroupThresholdHead(tail_reason_indices=TAIL_REASONS, label_delta_max=1.0)
    action = torch.tensor([[0.2, -0.1, 0.4, -0.3], [-0.2, 0.3, -0.5, 0.1]])
    reason = torch.zeros(2, 21)
    action_targets = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    reason_targets = torch.zeros(2, 21)
    reason_targets[:, 0] = torch.tensor([1.0, 0.0])
    output = head.calibration_objective(
        action,
        reason,
        action_targets,
        reason_targets,
        surrogate_temperature=0.20,
    )
    raw = torch.cat((action, reason), -1)
    targets = torch.cat((action_targets, reason_targets), -1)
    surrogate = (raw - head.compose_theta()) / 0.20
    probability = torch.sigmoid(surrogate)
    valid = targets.sum(0) > 0
    soft_f1 = 1.0 - (
        (2.0 * (probability * targets).sum(0) + 1e-8)
        / (probability.sum(0) + targets.sum(0) + 1e-8)
    )[valid].mean()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(surrogate, targets)
    rate = (probability.mean(0) - targets.mean(0)).square().mean()
    delta = head.label_delta.square().mean()
    card = torch.nn.functional.smooth_l1_loss(
        probability[:, :4].sum(-1), targets[:, :4].sum(-1), beta=1.0
    )
    expected = soft_f1 + 0.05 * bce + 0.02 * rate + 0.01 * delta + 0.02 * card
    assert torch.allclose(output["loss_calibration_total"], expected)
    assert output["soft_f1_valid_label_count"] == 5


def test_threshold_head_has_no_test_oracle_or_sample_feature_update_path() -> None:
    source = inspect.getsource(MOSAICGroupThresholdHead)
    assert "test_oracle" not in source
    assert "factor_credit" not in source
    assert "posterior" not in source
    assert "propensity" not in source
