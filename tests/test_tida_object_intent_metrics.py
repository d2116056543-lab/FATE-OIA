import torch

from fate_oia.utils.tida_object_intent_metrics import (
    apply_object_intent_utility_policy_to_rows,
    fit_object_intent_deployment_gates,
    fit_object_intent_utility_policy_oof,
    object_intent_traffic_metrics,
)


def _rows(samples: int, tracks: int) -> dict[str, torch.Tensor]:
    action_target = torch.randint(0, 2, (samples, 4)).float()
    reason_target = torch.randint(0, 2, (samples, 21)).float()
    action_attention = torch.softmax(torch.randn(samples, 4, tracks), -1)
    reason_attention = torch.softmax(torch.randn(samples, 21, tracks), -1)
    action_delta = 0.02 * (2.0 * action_target - 1.0)
    reason_delta = 0.02 * (2.0 * reason_target - 1.0)
    return {
        "pre_object_intent_action": torch.randn(samples, 4),
        "pre_object_intent_reason": torch.randn(samples, 21),
        "video_action": torch.randn(samples, 4) + action_delta,
        "video_reason": torch.randn(samples, 21) + reason_delta,
        "action_target": action_target,
        "reason_target": reason_target,
        "object_intent_action_delta": action_delta,
        "object_intent_reason_delta": reason_delta,
        "object_intent_action_selected_deleted_delta": torch.zeros_like(action_delta),
        "object_intent_action_control_deleted_delta": 0.5 * action_delta,
        "object_intent_reason_selected_deleted_delta": torch.zeros_like(reason_delta),
        "object_intent_reason_control_deleted_delta": 0.5 * reason_delta,
        "object_intent_action_support": torch.ones(samples, 4),
        "object_intent_reason_support": torch.ones(samples, 21),
        "object_intent_action_attention": action_attention,
        "object_intent_reason_attention": reason_attention,
        "object_intent_action_semantic_attention": action_attention,
        "object_intent_action_motion_attention": action_attention.roll(1, -1),
        "object_intent_reason_semantic_attention": reason_attention,
        "object_intent_reason_motion_attention": reason_attention.roll(1, -1),
        "object_intent_action_motion_mix": torch.full((samples, 4), 0.5),
        "object_intent_reason_motion_mix": torch.full((samples, 21), 0.5),
        "object_intent_interaction_risk": torch.linspace(0.0, 1.0, samples)[:, None].expand(-1, tracks),
        "object_intent_future_approach_risk": torch.linspace(
            0.0, 1.0, samples
        )[:, None].expand(-1, tracks),
    }


def test_object_metrics_report_temporal_identity_and_complexity_strata():
    samples, tracks = 12, 3
    rows = _rows(samples=samples, tracks=tracks)
    rows["object_intent_track_role_consistency"] = torch.linspace(
        0.2, 0.9, samples * tracks
    ).reshape(samples, tracks)
    weights = torch.rand(samples, 4, tracks)
    rows["object_intent_semantic_temporal_weights"] = weights / weights.sum(1, keepdim=True)
    metrics = object_intent_traffic_metrics(
        rows, torch.full((25,), 0.5), bootstrap_samples=16
    )
    assert 0.0 <= metrics["temporal_identity"]["role_consistency_mean"] <= 1.0
    assert metrics["temporal_identity"]["effective_visible_frames_mean"] >= 1.0
    assert set(metrics["traffic_complexity_strata"]) == {"low", "medium", "high"}
    assert sum(row["samples"] for row in metrics["traffic_complexity_strata"].values()) == samples
    assert len(metrics["future_approach_effectiveness"]["quartiles"]) == 4
    assert "critical_net_corrections" in metrics["future_approach_effectiveness"]["action"]


def test_object_intent_metrics_measure_prediction_utility_and_causal_route_effectiveness():
    samples, tracks = 8, 5
    action_target = torch.tensor([[1.0, 0.0, 1.0, 0.0]]).expand(samples, -1)
    reason_target = torch.tensor([[1.0, 0.0, 1.0]]).expand(samples, -1)
    action_sign = 2.0 * action_target - 1.0
    reason_sign = 2.0 * reason_target - 1.0
    action_delta = 0.20 * action_sign
    reason_delta = 0.15 * reason_sign
    action_attention = torch.zeros(samples, 4, tracks)
    reason_attention = torch.zeros(samples, 3, tracks)
    action_attention[..., 0] = 1.0
    reason_attention[..., 0] = 1.0
    rows = {
        "pre_object_intent_action": torch.zeros(samples, 4),
        "pre_object_intent_reason": torch.zeros(samples, 3),
        "video_action": action_delta,
        "video_reason": reason_delta,
        "action_target": action_target,
        "reason_target": reason_target,
        "object_intent_action_delta": action_delta,
        "object_intent_reason_delta": reason_delta,
        "object_intent_action_selected_deleted_delta": torch.zeros_like(action_delta),
        "object_intent_action_control_deleted_delta": 0.75 * action_delta,
        "object_intent_reason_selected_deleted_delta": torch.zeros_like(reason_delta),
        "object_intent_reason_control_deleted_delta": 0.75 * reason_delta,
        "object_intent_action_support": torch.ones(samples, 4),
        "object_intent_reason_support": torch.ones(samples, 3),
        "object_intent_action_attention": action_attention,
        "object_intent_reason_attention": reason_attention,
        "object_intent_action_semantic_attention": action_attention.roll(1, dims=-1),
        "object_intent_action_motion_attention": action_attention,
        "object_intent_reason_semantic_attention": reason_attention.roll(1, dims=-1),
        "object_intent_reason_motion_attention": reason_attention,
        "object_intent_action_motion_mix": torch.full((samples, 4), 0.75),
        "object_intent_reason_motion_mix": torch.full((samples, 3), 0.60),
        "object_intent_interaction_risk": torch.linspace(0.0, 1.0, samples)[:, None].expand(-1, tracks),
        "object_intent_future_approach_risk": torch.linspace(
            0.0, 1.0, samples
        )[:, None].expand(-1, tracks),
    }
    action_pair_attention = torch.zeros(samples, 4, tracks, tracks)
    reason_pair_attention = torch.zeros(samples, 3, tracks, tracks)
    action_pair_attention[:, :, 0, 1] = 1.0
    reason_pair_attention[:, :, 0, 1] = 1.0
    pair_min_distance = torch.ones(samples, tracks, tracks)
    pair_min_distance[:, 0, 1] = 0.1
    pair_reduction = torch.zeros(samples, tracks, tracks)
    pair_reduction[:, 0, 1] = 0.5
    rows.update({
        "object_intent_action_pair_attention": action_pair_attention,
        "object_intent_reason_pair_attention": reason_pair_attention,
        "object_intent_action_pair_candidate": 0.5 * action_delta,
        "object_intent_reason_pair_candidate": 0.5 * reason_delta,
        "object_intent_action_candidate": action_delta,
        "object_intent_reason_candidate": reason_delta,
        "object_intent_action_selected_pair_deleted_candidate": torch.zeros_like(action_delta),
        "object_intent_action_control_pair_deleted_candidate": 0.75 * action_delta,
        "object_intent_reason_selected_pair_deleted_candidate": torch.zeros_like(reason_delta),
        "object_intent_reason_control_pair_deleted_candidate": 0.75 * reason_delta,
        "object_intent_action_deploy_gate": torch.ones_like(action_delta),
        "object_intent_reason_deploy_gate": torch.ones_like(reason_delta),
        "object_intent_pair_min_future_distance": pair_min_distance,
        "object_intent_pair_distance_reduction": pair_reduction,
        "object_intent_action_utility_gate": torch.full_like(action_delta, 0.9),
        "object_intent_reason_utility_gate": torch.full_like(reason_delta, 0.9),
        "object_intent_action_utility_selected": torch.ones_like(action_delta),
        "object_intent_reason_utility_selected": torch.ones_like(reason_delta),
        "object_intent_action_deploy_scale": torch.ones_like(action_delta),
        "object_intent_reason_deploy_scale": torch.ones_like(reason_delta),
    })

    result = object_intent_traffic_metrics(rows, torch.full((7,), 0.5), bootstrap_samples=50)

    assert result["action"]["conditional_information_gain_bits"] > 0
    assert result["reason"]["conditional_information_gain_bits"] > 0
    assert result["action"]["selected_minus_random_deletion_gap"] > 0
    assert result["action"]["target_effective_route_rate"] == 1.0
    assert result["action"]["net_corrected_per_1000_labels"] > 0
    assert result["action"]["motion_semantic_attention_jsd"] > 0
    assert result["action"]["motion_semantic_selected_track_disagreement_rate"] == 1.0
    assert result["action"]["motion_mix_mean"] == 0.75
    assert result["action"]["future_pair_interaction"]["pair_transport_precision"] == 1.0
    assert result["action"]["future_pair_interaction"][
        "selected_minus_control_pair_deletion_gap"
    ] > 0
    assert result["action"]["future_pair_interaction"][
        "selected_pair_min_future_distance_mean"
    ] < 0.2
    assert len(result["interaction_risk_quartiles"]) == 4
    assert result["action"]["utility_quality"]["selected_benefit_rate"] == 1.0
    assert len(result["action"]["utility_quality"]["coverage_risk_curve"]) == 4


def test_fit_deployment_gates_opens_only_proper_score_improvements():
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    base = torch.tensor([[0.2, -1.0], [0.1, -0.8], [-0.2, 0.8], [-0.1, 1.0]])
    candidate = torch.tensor([[0.8, 0.8], [0.7, 0.7], [-0.8, -0.7], [-0.7, -0.8]])

    fitted = fit_object_intent_deployment_gates(
        base, candidate, target, min_samples=2, min_nll_improvement=1e-4
    )

    assert fitted["gate"].tolist() == [1.0, 0.0]
    assert fitted["nll_improvement"][0] > 0
    assert fitted["nll_improvement"][1] < 0


def test_deployment_gate_rejects_proper_score_gain_that_harms_locked_f1():
    target = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
    base = torch.tensor([[0.90], [0.90], [0.90], [0.80]])
    candidate = torch.full_like(base, 0.10)
    threshold = torch.tensor([0.70])

    fitted = fit_object_intent_deployment_gates(
        base,
        candidate,
        target,
        min_samples=1,
        min_nll_improvement=0.0,
        thresholds=threshold,
        min_f1_improvement=0.0,
    )

    assert fitted["nll_improvement"][0] > 0
    assert fitted["brier_improvement"][0] > 0
    assert fitted["fixed_threshold_f1_improvement"][0] < 0
    assert fitted["gate"].tolist() == [0.0]


def test_utility_policy_oof_selects_helpful_route_and_keeps_zero_fallback():
    torch.manual_seed(9)
    samples = 120
    target = torch.randint(0, 2, (samples, 2)).float()
    sign = 2.0 * target - 1.0
    base = 0.25 * torch.randn(samples, 2)
    candidate = torch.stack((0.03 * sign[:, 0], -0.03 * sign[:, 1]), dim=1)
    utility = torch.stack((0.9 * torch.ones(samples), 0.1 * torch.ones(samples)), dim=1)
    threshold = torch.full((2,), 0.5)

    policy = fit_object_intent_utility_policy_oof(
        base, candidate, utility, target, threshold,
        scales=(0.0, 8.0, 16.0, 32.0, 64.0),
        cutoffs=(0.0, 0.5), folds=5,
    )

    assert policy["scale"][0] > 0
    assert policy["scale"][1] == 0
    assert policy["oof_gain"][0] > 0
    assert policy["oof_gain"][1] == 0


def test_utility_policy_respects_coverage_and_benefit_precision_guards():
    samples = 120
    target = torch.zeros(samples, 1)
    base = torch.zeros(samples, 1)
    candidate = torch.full((samples, 1), 0.02)
    utility = torch.linspace(0.0, 1.0, samples)[:, None]

    policy = fit_object_intent_utility_policy_oof(
        base, candidate, utility, target, torch.tensor([0.5]),
        scales=(0.0, 16.0), cutoffs=(0.0,), folds=5,
        max_selected_rate=0.5, min_selected_benefit_rate=0.75,
    )

    assert policy["scale"].tolist() == [0.0]
    assert policy["selected_rate"].tolist() == [0.0]


def test_utility_policy_requires_cross_fold_stability():
    samples = 100
    generator = torch.Generator(device="cpu").manual_seed(3407)
    permutation = torch.randperm(samples, generator=generator)
    fold_ids = torch.empty(samples, dtype=torch.long)
    fold_ids[permutation] = torch.arange(samples) % 5
    target = (torch.arange(samples) % 2).float()[:, None]
    sign = 2.0 * target - 1.0
    base = 0.04 * sign
    candidate = torch.zeros(samples, 1)
    utility = torch.ones(samples, 1)
    # Correct one deliberately flipped row in only three of five held-out folds.
    # The mean gain is positive but is not reproducible enough for deployment.
    for fold in range(3):
        row = int((fold_ids == fold).nonzero()[0])
        base[row] = -0.04 * sign[row]
        candidate[row] = 0.02 * sign[row]

    policy = fit_object_intent_utility_policy_oof(
        base, candidate, utility, target, torch.tensor([0.5]),
        scales=(0.0, 16.0), cutoffs=(0.0,), folds=5,
        min_positive_fold_fraction=0.8, cap=0.08,
    )

    assert policy["scale"].tolist() == [0.0]
    assert policy["positive_fold_fraction"].tolist() == [0.0]


def test_inactive_utility_policy_is_not_reported_as_selected():
    rows = {
        "pre_object_intent_action": torch.zeros(4, 1),
        "pre_object_intent_reason": torch.zeros(4, 1),
        "object_intent_action_candidate": torch.ones(4, 1),
        "object_intent_reason_candidate": torch.ones(4, 1),
        "object_intent_action_utility_gate": torch.ones(4, 1),
        "object_intent_reason_utility_gate": torch.ones(4, 1),
    }
    closed = {
        "gate": torch.zeros(1), "scale": torch.zeros(1), "cutoff": torch.zeros(1),
    }

    applied = apply_object_intent_utility_policy_to_rows(rows, closed, closed)

    assert not applied["object_intent_action_utility_selected"].bool().any()
    assert not applied["object_intent_reason_utility_selected"].bool().any()
