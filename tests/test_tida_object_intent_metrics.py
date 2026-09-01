import torch

from fate_oia.utils.tida_object_intent_metrics import (
    apply_object_intent_utility_policy_to_rows,
    combine_object_intent_utility_policies,
    concatenate_object_intent_policy_rows,
    fit_object_intent_deployment_gates,
    fit_object_intent_utility_policy_oof,
    object_intent_traffic_metrics,
    object_intent_policy_fold_groups,
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
        "source_batches": ["batch_a"] * 4 + ["batch_b"] * 4,
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
    source_metrics = result["source_generalization"]["action"]
    assert source_metrics["source_count"] == 2
    assert source_metrics["positive_information_gain_source_fraction"] == 1.0
    assert source_metrics["worst_source_information_gain_bits"] > 0


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
    target = (torch.arange(samples) % 2).float()[:, None]
    generator = torch.Generator(device="cpu").manual_seed(3407)
    fold_ids = torch.empty(samples, dtype=torch.long)
    for membership in (target[:, 0] > 0.5, target[:, 0] <= 0.5):
        rows = membership.nonzero(as_tuple=False).flatten()
        order = rows[torch.randperm(rows.numel(), generator=generator)]
        fold_ids[order] = torch.arange(rows.numel()) % 5
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
        min_selected_benefit_rate=0.0,
        min_positive_fold_fraction=0.8, cap=0.08,
    )

    assert policy["scale"].tolist() == [0.0]
    assert policy["positive_fold_fraction"].tolist() == [0.0]
    assert torch.allclose(
        policy["best_candidate_positive_fold_fraction"], torch.tensor([0.6])
    )


def test_utility_policy_accepts_repeatable_gain_with_quantized_neutral_folds():
    samples = 100
    target = (torch.arange(samples) % 2).float()[:, None]
    generator = torch.Generator(device="cpu").manual_seed(3407)
    fold_ids = torch.empty(samples, dtype=torch.long)
    for membership in (target[:, 0] > 0.5, target[:, 0] <= 0.5):
        rows = membership.nonzero(as_tuple=False).flatten()
        order = rows[torch.randperm(rows.numel(), generator=generator)]
        fold_ids[order] = torch.arange(rows.numel()) % 5
    sign = 2.0 * target - 1.0
    base = 0.04 * sign
    candidate = torch.zeros(samples, 1)
    utility = torch.ones(samples, 1)
    for fold in range(3):
        row = int((fold_ids == fold).nonzero()[0])
        base[row] = -0.04 * sign[row]
        candidate[row] = 0.02 * sign[row]

    policy = fit_object_intent_utility_policy_oof(
        base, candidate, utility, target, torch.tensor([0.5]),
        scales=(0.0, 16.0), cutoffs=(0.0,), folds=5,
        min_selected_benefit_rate=0.0,
        min_positive_fold_fraction=0.4,
        min_non_degrading_fold_fraction=0.8,
        fold_degradation_tolerance=0.002,
        cap=0.08,
    )

    assert policy["scale"].tolist() == [16.0]
    assert torch.allclose(policy["positive_fold_fraction"], torch.tensor([0.6]))
    assert policy["non_degrading_fold_fraction"].tolist() == [1.0]


def test_utility_policy_can_use_train_only_proper_score_tie_without_f1_harm():
    samples = 100
    target = (torch.arange(samples) % 2).float()[:, None]
    sign = 2.0 * target - 1.0
    base = sign.clone()
    candidate = 0.01 * sign
    utility = torch.ones(samples, 1)

    conservative = fit_object_intent_utility_policy_oof(
        base, candidate, utility, target, torch.tensor([0.5]),
        scales=(0.0, 4.0), cutoffs=(0.0,), folds=5,
        min_selected_benefit_rate=0.65,
        min_non_degrading_fold_fraction=1.0,
    )
    proper_tie = fit_object_intent_utility_policy_oof(
        base, candidate, utility, target, torch.tensor([0.5]),
        scales=(0.0, 4.0), cutoffs=(0.0,), folds=5,
        min_selected_benefit_rate=0.65,
        min_non_degrading_fold_fraction=1.0,
        allow_proper_score_tie=True,
    )

    assert conservative["scale"].tolist() == [0.0]
    assert proper_tie["scale"].tolist() == [4.0]
    assert proper_tie["proper_score_tie_selected"].tolist() == [1.0]
    assert proper_tie["nll_improvement"][0] > 0
    assert proper_tie["brier_improvement"][0] > 0


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


def test_dual_utility_policy_selects_best_source_per_label():
    directional = {
        "gate": torch.tensor([1.0, 0.0]),
        "scale": torch.tensor([8.0, 0.0]),
        "cutoff": torch.tensor([0.5, 0.0]),
        "oof_gain": torch.tensor([0.03, 0.0]),
    }
    risk = {
        "gate": torch.tensor([0.0, 1.0]),
        "scale": torch.tensor([0.0, 16.0]),
        "cutoff": torch.tensor([0.0, 0.6]),
        "oof_gain": torch.tensor([0.0, 0.04]),
    }

    combined = combine_object_intent_utility_policies(directional, risk)

    assert combined["utility_source"].tolist() == [0, 1]
    assert combined["scale"].tolist() == [8.0, 16.0]
    torch.testing.assert_close(combined["oof_gain"], torch.tensor([0.03, 0.04]))


def test_apply_dual_policy_uses_selected_utility_without_test_labels():
    rows = {
        "pre_object_intent_action": torch.zeros(2, 2),
        "pre_object_intent_reason": torch.zeros(2, 2),
        "object_intent_action_candidate": torch.full((2, 2), 0.01),
        "object_intent_reason_candidate": torch.full((2, 2), 0.01),
        "object_intent_action_utility_gate": torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
        "object_intent_reason_utility_gate": torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
        "object_intent_action_risk_utility_gate": torch.tensor([[0.1, 0.9], [0.1, 0.9]]),
        "object_intent_reason_risk_utility_gate": torch.tensor([[0.1, 0.9], [0.1, 0.9]]),
    }
    policy = {
        "gate": torch.ones(2),
        "scale": torch.ones(2),
        "cutoff": torch.full((2,), 0.5),
        "utility_source": torch.tensor([0, 1]),
    }

    applied = apply_object_intent_utility_policy_to_rows(rows, policy, policy)

    assert applied["object_intent_action_utility_selected"].all()
    assert applied["object_intent_action_utility_source"][0].tolist() == [0, 1]


def test_apply_dual_policy_accepts_directional_risk_artifact_without_legacy_gate():
    rows = {
        "pre_object_intent_action": torch.zeros(2, 1),
        "pre_object_intent_reason": torch.zeros(2, 1),
        "object_intent_action_candidate": torch.full((2, 1), 0.01),
        "object_intent_reason_candidate": torch.full((2, 1), 0.01),
        "object_intent_action_directional_utility_gate": torch.full((2, 1), 0.9),
        "object_intent_reason_directional_utility_gate": torch.full((2, 1), 0.9),
        "object_intent_action_risk_utility_gate": torch.full((2, 1), 0.8),
        "object_intent_reason_risk_utility_gate": torch.full((2, 1), 0.8),
    }
    policy = {
        "gate": torch.ones(1),
        "scale": torch.ones(1),
        "cutoff": torch.full((1,), 0.5),
        "utility_source": torch.ones(1, dtype=torch.long),
    }

    applied = apply_object_intent_utility_policy_to_rows(rows, policy, policy)

    assert applied["object_intent_action_utility_selected"].all()


def test_dual_policy_uses_proper_score_to_break_neutral_f1_tie():
    directional = {
        "gate": torch.tensor([1.0]),
        "scale": torch.tensor([8.0]),
        "cutoff": torch.tensor([0.5]),
        "oof_gain": torch.tensor([0.0]),
        "nll_improvement": torch.tensor([0.001]),
        "brier_improvement": torch.tensor([0.001]),
    }
    risk = {
        "gate": torch.tensor([1.0]),
        "scale": torch.tensor([16.0]),
        "cutoff": torch.tensor([0.6]),
        "oof_gain": torch.tensor([0.0]),
        "nll_improvement": torch.tensor([0.004]),
        "brier_improvement": torch.tensor([0.003]),
    }

    combined = combine_object_intent_utility_policies(directional, risk)

    assert combined["utility_source"].item() == 1
    assert combined["scale"].item() == 16.0
    assert combined["risk_proper_gain"].item() > combined["directional_proper_gain"].item()


def test_policy_rows_concatenate_only_train_cohorts_and_preserve_order():
    keys = (
        "pre_object_intent_action", "pre_object_intent_reason",
        "object_intent_action_candidate", "object_intent_reason_candidate",
        "object_intent_action_directional_utility_gate",
        "object_intent_reason_directional_utility_gate",
        "object_intent_action_risk_utility_gate",
        "object_intent_reason_risk_utility_gate",
        "action_target", "reason_target",
    )
    calib = {key: torch.zeros(3, 2) for key in keys}
    audit = {key: torch.ones(4, 2) for key in keys}
    calib["source_batches"] = ["batch1", "batch1", "batch2"]
    calib["source_video_ids"] = ["cv0", "cv1", "cv2"]
    calib["file_names"] = ["c0.mp4", "c1.mp4", "c2.mp4"]
    audit["source_batches"] = ["batch3"] * 4
    audit["source_video_ids"] = [f"av{index}" for index in range(4)]
    audit["file_names"] = [f"a{index}.mp4" for index in range(4)]

    combined = concatenate_object_intent_policy_rows(
        (("train_calib", calib), ("train_audit", audit))
    )

    assert combined["action_target"].shape == (7, 2)
    assert combined["action_target"][:3].eq(0).all()
    assert combined["action_target"][3:].eq(1).all()
    assert combined["_policy_cohort_sizes"] == {"train_calib": 3, "train_audit": 4}
    assert combined["source_batches"] == [
        "batch1", "batch1", "batch2", "batch3", "batch3", "batch3", "batch3",
    ]
    assert combined["file_names"] == [
        "c0.mp4", "c1.mp4", "c2.mp4", "a0.mp4", "a1.mp4", "a2.mp4", "a3.mp4",
    ]
    assert combined["source_video_ids"] == [
        "cv0", "cv1", "cv2", "av0", "av1", "av2", "av3",
    ]


def test_policy_fold_groups_fall_back_to_video_identity_inside_one_domain():
    rows = {
        "source_batches": ["one_domain"] * 4,
        "source_video_ids": ["video_a", "video_a", "video_b", "video_c"],
    }

    group_ids, strategy = object_intent_policy_fold_groups(rows)

    assert strategy == "leave_source_video_out"
    assert group_ids.tolist() == [0, 0, 1, 2]


def test_policy_fold_groups_prefer_domain_when_multiple_domains_exist():
    rows = {
        "source_batches": ["domain_a", "domain_a", "domain_b"],
        "source_video_ids": ["video_a", "video_b", "video_c"],
    }

    group_ids, strategy = object_intent_policy_fold_groups(rows)

    assert strategy == "leave_source_batch_out"
    assert group_ids.tolist() == [0, 0, 1]


def test_utility_policy_leave_source_out_rejects_large_source_spurious_gain():
    large, small = 100, 10
    source_groups = torch.cat((
        torch.zeros(large, dtype=torch.long),
        torch.ones(small, dtype=torch.long),
        torch.full((small,), 2, dtype=torch.long),
    ))
    target = torch.cat((
        torch.ones(large, 1), torch.zeros(small * 2, 1),
    ))
    base = torch.cat((
        torch.full((large, 1), -0.02), torch.full((small * 2, 1), -0.02),
    ))
    candidate = torch.full_like(base, 0.02)
    utility = torch.ones_like(base)

    random_oof = fit_object_intent_utility_policy_oof(
        base, candidate, utility, target, torch.tensor([0.5]),
        scales=(0.0, 4.0), cutoffs=(0.0,), folds=3,
        min_selected_benefit_rate=0.0,
    )
    source_oof = fit_object_intent_utility_policy_oof(
        base, candidate, utility, target, torch.tensor([0.5]),
        scales=(0.0, 4.0), cutoffs=(0.0,), folds=3,
        min_selected_benefit_rate=0.0,
        min_positive_fold_fraction=0.6,
        fold_group_ids=source_groups,
    )

    assert random_oof["scale"].item() > 0
    assert source_oof["scale"].item() == 0
    assert source_oof["fold_strategy"] == "group_kfold"
    assert source_oof["fold_count"] == 3
    assert source_oof["source_group_count"] == 3


def test_utility_policy_caps_many_source_groups_to_requested_group_folds():
    samples = 24
    groups = torch.arange(samples) // 2
    target = (torch.arange(samples) % 3 == 0).float()[:, None]
    base = torch.zeros(samples, 1)
    candidate = (2.0 * target - 1.0) * 0.01

    policy = fit_object_intent_utility_policy_oof(
        base, candidate, torch.ones_like(base), target, torch.tensor([0.5]),
        scales=(0.0, 1.0), cutoffs=(0.0,), folds=5,
        min_selected_benefit_rate=0.0, fold_group_ids=groups,
    )

    assert policy["source_group_count"] == 12
    assert policy["fold_count"] == 5
    assert policy["fold_strategy"] == "group_kfold"


def test_policy_rows_reject_test_or_oracle_cohorts():
    rows = {
        "pre_object_intent_action": torch.zeros(2, 1),
        "pre_object_intent_reason": torch.zeros(2, 1),
        "object_intent_action_candidate": torch.zeros(2, 1),
        "object_intent_reason_candidate": torch.zeros(2, 1),
        "object_intent_action_directional_utility_gate": torch.zeros(2, 1),
        "object_intent_reason_directional_utility_gate": torch.zeros(2, 1),
        "object_intent_action_risk_utility_gate": torch.zeros(2, 1),
        "object_intent_reason_risk_utility_gate": torch.zeros(2, 1),
        "action_target": torch.zeros(2, 1),
        "reason_target": torch.zeros(2, 1),
    }
    for forbidden in ("test", "test_oracle", "oracle"):
        try:
            concatenate_object_intent_policy_rows(((forbidden, rows),))
        except ValueError as error:
            assert "train-only" in str(error)
        else:
            raise AssertionError(f"forbidden policy cohort {forbidden} was accepted")
