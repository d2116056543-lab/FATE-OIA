import torch

from fate_oia.utils.tida_temporal_metrics import (
    paired_temporal_contribution,
    robust_motion_score,
)


def test_motion_score_is_label_independent_and_orders_stronger_motion():
    velocity = torch.tensor([[[0.1, 0.0]], [[1.0, 0.0]], [[4.0, 0.0]], [[8.0, 0.0]]])
    acceleration = torch.zeros_like(velocity)
    score = robust_motion_score(velocity, acceleration)
    assert score.shape == (4,)
    assert torch.all(score[1:] > score[:-1])


def test_temporal_contribution_reports_benefit_harm_and_deterministic_bootstrap():
    image = torch.zeros(8, 2)
    target = torch.tensor([[1.0, 0.0]] * 8)
    delta = torch.tensor([[0.2, -0.2]] * 6 + [[-0.1, 0.1]] * 2)
    kwargs = dict(
        image_logits=image,
        video_logits=image + delta,
        target=target,
        motion_score=torch.arange(8).float(),
        bootstrap_samples=200,
        seed=11,
    )
    first = paired_temporal_contribution(**kwargs)
    second = paired_temporal_contribution(**kwargs)
    assert first == second
    assert first["full"]["benefit_rate"] == 0.75
    assert first["full"]["harm_rate"] == 0.25
    assert first["full"]["signed_margin_mean"] > 0
    assert len(first["motion_quartiles"]) == 4


def test_slice_marks_labels_without_both_classes_unavailable():
    image = torch.zeros(4, 2)
    target = torch.tensor([[1.0, 0.0]] * 4)
    result = paired_temporal_contribution(
        image, image + 0.1, target, motion_score=torch.arange(4).float(),
        bootstrap_samples=20, seed=3,
    )
    assert result["full"]["eligible_label_count"] == 0
    assert all(not row["class_metrics_available"] for row in result["per_label"])


def test_reason_pu_weight_does_not_treat_every_observed_zero_as_hard_negative():
    image = torch.zeros(2, 2)
    video = torch.tensor([[0.2, 0.2], [0.2, 0.2]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    hard = paired_temporal_contribution(
        image, video, target, motion_score=torch.arange(2).float(),
        bootstrap_samples=20, seed=2,
    )
    pu = paired_temporal_contribution(
        image, video, target, motion_score=torch.arange(2).float(),
        pu_negative_weight=torch.zeros_like(target), bootstrap_samples=20, seed=2,
    )
    assert hard["full"]["signed_margin_mean"] == 0.0
    assert pu["full"]["signed_margin_mean"] > 0.19
    assert pu["full"]["observed_positive_margin_mean"] > 0
    assert pu["full"]["observed_zero_logit_delta_mean"] > 0
