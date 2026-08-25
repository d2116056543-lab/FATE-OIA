from __future__ import annotations

import torch
from torch.nn import functional as F

from .tida_relational_traffic_metrics import relational_traffic_metrics

ROLE_NAMES = ("background", "vehicle", "vulnerable_road_user", "traffic_control", "lane_drivable")


def _average_precision(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    values = []
    for label in range(logits.shape[1]):
        order = logits[:, label].argsort(descending=True)
        truth = target[order, label].float()
        positives = truth.sum()
        if positives <= 0:
            values.append(logits.new_tensor(float("nan")))
            continue
        precision = truth.cumsum(0) / torch.arange(
            1, truth.numel() + 1, device=truth.device, dtype=truth.dtype
        )
        values.append((precision * truth).sum() / positives)
    return torch.stack(values)


def _fixed_macro_metrics(
    logits: torch.Tensor, target: torch.Tensor, thresholds: torch.Tensor
) -> tuple[float, float]:
    prediction = logits.sigmoid() >= thresholds[None]
    truth = target > 0.5
    true_positive = (prediction & truth).sum(0).float()
    false_positive = (prediction & ~truth).sum(0).float()
    false_negative = (~prediction & truth).sum(0).float()
    f1 = 2.0 * true_positive / (
        2.0 * true_positive + false_positive + false_negative
    ).clamp_min(1.0)
    average_precision = _average_precision(logits, target)
    return float(f1.mean()), float(average_precision[torch.isfinite(average_precision)].mean())


def _traffic_complexity_strata(
    rows: dict[str, torch.Tensor], thresholds: torch.Tensor
) -> dict[str, dict[str, float | int]]:
    action_count = rows["action_target"].shape[1]
    sample_risk = rows["object_intent_interaction_risk"].amax(-1)
    order = sample_risk.argsort()
    result: dict[str, dict[str, float | int]] = {}
    for name, indices in zip(("low", "medium", "high"), torch.tensor_split(order, 3)):
        action_base = _fixed_macro_metrics(
            rows["pre_object_intent_action"][indices], rows["action_target"][indices],
            thresholds[:action_count],
        )
        action_final = _fixed_macro_metrics(
            rows["video_action"][indices], rows["action_target"][indices],
            thresholds[:action_count],
        )
        reason_base = _fixed_macro_metrics(
            rows["pre_object_intent_reason"][indices], rows["reason_target"][indices],
            thresholds[action_count:],
        )
        reason_final = _fixed_macro_metrics(
            rows["video_reason"][indices], rows["reason_target"][indices],
            thresholds[action_count:],
        )
        result[name] = {
            "samples": int(indices.numel()),
            "interaction_risk_mean": float(sample_risk[indices].mean()),
            "action_base_mf1": action_base[0],
            "action_final_mf1": action_final[0],
            "action_mf1_gain": action_final[0] - action_base[0],
            "action_base_map": action_base[1],
            "action_final_map": action_final[1],
            "action_map_gain": action_final[1] - action_base[1],
            "reason_base_mf1": reason_base[0],
            "reason_final_mf1": reason_final[0],
            "reason_mf1_gain": reason_final[0] - reason_base[0],
            "reason_base_map": reason_base[1],
            "reason_final_map": reason_final[1],
            "reason_map_gain": reason_final[1] - reason_base[1],
        }
    return result


def _rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return float("nan")
    left_rank = left.argsort().argsort().float()
    right_rank = right.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack((left_rank, right_rank)))[0, 1])


def _future_approach_effectiveness(
    rows: dict[str, torch.Tensor], thresholds: torch.Tensor,
) -> dict[str, object]:
    approach = rows["object_intent_future_approach_risk"].float()
    sample_score = approach.amax(-1)
    action_count = rows["action_target"].shape[1]
    result: dict[str, object] = {
        "sample_score_mean": float(sample_score.mean()),
        "sample_score_p50": float(sample_score.quantile(0.5)),
        "sample_score_p90": float(sample_score.quantile(0.9)),
        "track_positive_rate": float((approach > 0).float().mean()),
    }
    for branch, count in (("action", action_count), ("reason", rows["reason_target"].shape[1])):
        base = rows[f"pre_object_intent_{branch}"]
        final = rows[f"video_{branch}"]
        target = rows[f"{branch}_target"].float()
        branch_thresholds = thresholds[:count] if branch == "action" else thresholds[action_count:]
        attention = rows[f"object_intent_{branch}_attention"]
        selected_approach = torch.einsum("nlk,nk->nl", attention, approach)
        signed_gain = ((2.0 * target - 1.0) * (final - base)).mean(-1)
        cutoff = sample_score.quantile(0.75)
        critical = sample_score >= cutoff
        base_prediction = base.sigmoid() >= branch_thresholds[None]
        final_prediction = final.sigmoid() >= branch_thresholds[None]
        truth = target > 0.5
        recovered = (~(base_prediction == truth)) & (final_prediction == truth)
        damaged = (base_prediction == truth) & (~(final_prediction == truth))
        result[branch] = {
            "attention_weighted_approach_mean": float(selected_approach.mean()),
            "attention_approach_lift_over_track_mean": float(
                selected_approach.mean() - approach.mean()
            ),
            "approach_utility_spearman": _rank_correlation(sample_score, signed_gain),
            "critical_sample_count": int(critical.sum()),
            "critical_errors_recovered": int(recovered[critical].sum()),
            "critical_correct_damaged": int(damaged[critical].sum()),
            "critical_net_corrections": int(
                recovered[critical].sum() - damaged[critical].sum()
            ),
        }
    quartiles = []
    order = sample_score.argsort()
    for index, indices in enumerate(torch.tensor_split(order, 4)):
        action_base = _fixed_macro_metrics(
            rows["pre_object_intent_action"][indices], rows["action_target"][indices],
            thresholds[:action_count],
        )
        action_final = _fixed_macro_metrics(
            rows["video_action"][indices], rows["action_target"][indices],
            thresholds[:action_count],
        )
        reason_base = _fixed_macro_metrics(
            rows["pre_object_intent_reason"][indices], rows["reason_target"][indices],
            thresholds[action_count:],
        )
        reason_final = _fixed_macro_metrics(
            rows["video_reason"][indices], rows["reason_target"][indices],
            thresholds[action_count:],
        )
        quartiles.append({
            "quartile": index + 1,
            "samples": int(indices.numel()),
            "future_approach_mean": float(sample_score[indices].mean()),
            "action_mf1_gain": action_final[0] - action_base[0],
            "action_map_gain": action_final[1] - action_base[1],
            "reason_mf1_gain": reason_final[0] - reason_base[0],
            "reason_map_gain": reason_final[1] - reason_base[1],
        })
    result["quartiles"] = quartiles
    return result


def fit_object_intent_deployment_gates(
    base_logits: torch.Tensor,
    candidate_delta: torch.Tensor,
    target: torch.Tensor,
    *,
    min_samples: int = 64,
    min_nll_improvement: float = 1e-4,
    thresholds: torch.Tensor | None = None,
    min_f1_improvement: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Open only label routes that improve two proper scores on train-calib."""
    if base_logits.shape != candidate_delta.shape or base_logits.shape != target.shape:
        raise ValueError("base logits, candidate delta, and target must share [N,L]")
    if base_logits.ndim != 2:
        raise ValueError("object-intent gate fitting expects [N,L]")
    base_nll = F.binary_cross_entropy_with_logits(base_logits, target.float(), reduction="none").mean(0)
    routed_nll = F.binary_cross_entropy_with_logits(
        base_logits + candidate_delta, target.float(), reduction="none"
    ).mean(0)
    base_brier = (base_logits.sigmoid() - target.float()).square().mean(0)
    routed_brier = ((base_logits + candidate_delta).sigmoid() - target.float()).square().mean(0)
    positive = target.sum(0)
    negative = target.shape[0] - positive
    enough = (positive >= min_samples) & (negative >= min_samples)
    nll_improvement = base_nll - routed_nll
    brier_improvement = base_brier - routed_brier
    if thresholds is None:
        f1_improvement = torch.zeros_like(nll_improvement)
        f1_safe = torch.ones_like(nll_improvement, dtype=torch.bool)
    else:
        thresholds = torch.as_tensor(
            thresholds, dtype=base_logits.dtype, device=base_logits.device
        )
        if thresholds.shape != (base_logits.shape[1],):
            raise ValueError("thresholds must contain one probability per label")

        def fixed_f1(logits: torch.Tensor) -> torch.Tensor:
            prediction = logits.sigmoid() >= thresholds[None]
            truth = target > 0.5
            true_positive = (prediction & truth).sum(0).float()
            false_positive = (prediction & ~truth).sum(0).float()
            false_negative = (~prediction & truth).sum(0).float()
            return 2.0 * true_positive / (
                2.0 * true_positive + false_positive + false_negative
            ).clamp_min(1.0)

        f1_improvement = fixed_f1(base_logits + candidate_delta) - fixed_f1(base_logits)
        f1_safe = f1_improvement >= float(min_f1_improvement)
    gate = (
        enough
        & (nll_improvement >= min_nll_improvement)
        & (brier_improvement >= 0)
        & f1_safe
    )
    return {
        "gate": gate.to(base_logits.dtype),
        "nll_improvement": nll_improvement,
        "brier_improvement": brier_improvement,
        "fixed_threshold_f1_improvement": f1_improvement,
        "positive_count": positive,
        "negative_count": negative,
    }


def fit_object_intent_utility_policy_oof(
    base_logits: torch.Tensor,
    candidate_delta: torch.Tensor,
    utility_gate: torch.Tensor,
    target: torch.Tensor,
    locked_thresholds: torch.Tensor,
    *,
    scales: tuple[float, ...] = (0.0, 4.0, 8.0, 16.0, 32.0, 64.0),
    cutoffs: tuple[float, ...] = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8),
    folds: int = 5,
    min_oof_gain: float = 0.0,
    max_selected_rate: float = 1.0,
    min_selected_benefit_rate: float = 0.5,
    min_nll_improvement: float = 0.0,
    min_brier_improvement: float = 0.0,
    quantile_coverages: tuple[float, ...] = (),
    cap: float = 0.08,
    seed: int = 3407,
) -> dict[str, torch.Tensor | list[dict[str, float]]]:
    """Fit selective route scale/cutoff using train-calib labels only."""
    if not (
        base_logits.shape == candidate_delta.shape == utility_gate.shape == target.shape
    ):
        raise ValueError("utility policy tensors must share [N,L]")
    if base_logits.ndim != 2 or base_logits.shape[0] < int(folds):
        raise ValueError("utility policy requires enough [N,L] rows for OOF fitting")
    locked_thresholds = torch.as_tensor(
        locked_thresholds, dtype=base_logits.dtype, device=base_logits.device
    )
    if locked_thresholds.shape != (base_logits.shape[1],):
        raise ValueError("locked_thresholds must contain one value per label")
    if 0.0 not in scales:
        raise ValueError("utility scales must include strict zero fallback")

    def label_f1(logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        prediction = logits.sigmoid() >= locked_thresholds[None]
        positive = truth > 0.5
        tp = (prediction & positive).sum(0).float()
        fp = (prediction & ~positive).sum(0).float()
        fn = (~prediction & positive).sum(0).float()
        return 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)

    cutoff_values = [float(cutoff) for cutoff in cutoffs]
    for coverage in quantile_coverages:
        if not 0.0 < float(coverage) <= 1.0:
            raise ValueError("utility quantile coverages must be in (0,1]")
        cutoff_values.extend(
            float(utility_gate[:, label].quantile(1.0 - float(coverage)))
            for label in range(utility_gate.shape[1])
        )
    cutoff_values = sorted(set(round(value, 7) for value in cutoff_values))
    candidates = [
        (float(scale), float(cutoff))
        for scale in scales for cutoff in cutoff_values
    ]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(base_logits.shape[0], generator=generator)
    fold_ids = torch.empty(base_logits.shape[0], dtype=torch.long)
    fold_ids[permutation] = torch.arange(base_logits.shape[0]) % int(folds)
    fold_ids = fold_ids.to(base_logits.device)
    scores = base_logits.new_zeros((len(candidates), base_logits.shape[1]))
    for fold in range(int(folds)):
        holdout = fold_ids == fold
        for index, (scale, cutoff) in enumerate(candidates):
            selected = utility_gate[holdout] >= cutoff
            delta = (float(scale) * candidate_delta[holdout]).clamp(-float(cap), float(cap))
            scores[index] += label_f1(
                base_logits[holdout] + selected.to(delta.dtype) * delta,
                target[holdout],
            )
    scores /= float(folds)
    candidate_selected_rate = base_logits.new_zeros(scores.shape)
    candidate_benefit_rate = base_logits.new_zeros(scores.shape)
    candidate_nll_improvement = base_logits.new_zeros(scores.shape)
    candidate_brier_improvement = base_logits.new_zeros(scores.shape)
    base_nll = F.binary_cross_entropy_with_logits(
        base_logits, target.float(), reduction="none"
    ).mean(0)
    base_brier = (base_logits.sigmoid() - target.float()).square().mean(0)
    sign = 2.0 * target.float() - 1.0
    for index, (scale, cutoff) in enumerate(candidates):
        if scale == 0.0:
            candidate_benefit_rate[index].fill_(1.0)
            continue
        selected = utility_gate >= cutoff
        delta = (float(scale) * candidate_delta).clamp(-float(cap), float(cap))
        deployed_delta = selected.to(delta.dtype) * delta
        selected_count = selected.sum(0).clamp_min(1)
        candidate_selected_rate[index] = selected.float().mean(0)
        candidate_benefit_rate[index] = (
            ((sign * delta) > 0) & selected
        ).sum(0).float() / selected_count
        routed = base_logits + deployed_delta
        candidate_nll_improvement[index] = base_nll - F.binary_cross_entropy_with_logits(
            routed, target.float(), reduction="none"
        ).mean(0)
        candidate_brier_improvement[index] = base_brier - (
            routed.sigmoid() - target.float()
        ).square().mean(0)
    zero_indices = [i for i, (scale, _) in enumerate(candidates) if scale == 0.0]
    zero = zero_indices[0]
    selected_indices, gains = [], []
    for label in range(base_logits.shape[1]):
        eligible = [
            index for index, (scale, _) in enumerate(candidates)
            if scale == 0.0 or (
                float(candidate_selected_rate[index, label]) <= float(max_selected_rate)
                and float(candidate_benefit_rate[index, label]) >= float(min_selected_benefit_rate)
                and float(candidate_nll_improvement[index, label]) >= float(min_nll_improvement)
                and float(candidate_brier_improvement[index, label]) >= float(min_brier_improvement)
            )
        ]
        best = max(
            eligible,
            key=lambda index: (
                float(scores[index, label]),
                -abs(candidates[index][0]),
                candidates[index][1],
            ),
        )
        gain = scores[best, label] - scores[zero, label]
        if float(gain) <= float(min_oof_gain):
            best = zero
            gain = gain.new_zeros(())
        selected_indices.append(best)
        gains.append(gain)
    policy = [candidates[index] for index in selected_indices]
    return {
        "gate": base_logits.new_tensor([float(scale != 0.0) for scale, _ in policy]),
        "scale": base_logits.new_tensor([scale for scale, _ in policy]),
        "cutoff": base_logits.new_tensor([cutoff for _, cutoff in policy]),
        "oof_gain": torch.stack(gains),
        "oof_scores": scores,
        "selected_rate": torch.stack([
            candidate_selected_rate[index, label]
            for label, index in enumerate(selected_indices)
        ]),
        "selected_benefit_rate": torch.stack([
            candidate_benefit_rate[index, label]
            for label, index in enumerate(selected_indices)
        ]),
        "nll_improvement": torch.stack([
            candidate_nll_improvement[index, label]
            for label, index in enumerate(selected_indices)
        ]),
        "brier_improvement": torch.stack([
            candidate_brier_improvement[index, label]
            for label, index in enumerate(selected_indices)
        ]),
        "candidates": [
            {"scale": scale, "cutoff": cutoff} for scale, cutoff in candidates
        ],
    }


def fit_object_intent_gates_from_rows(
    rows: dict[str, torch.Tensor],
    *,
    min_samples: int = 64,
    min_nll_improvement: float = 1e-4,
    thresholds: torch.Tensor | None = None,
    min_f1_improvement: float = 0.0,
) -> dict[str, dict[str, torch.Tensor]]:
    action_count = rows["action_target"].shape[1]
    reason_count = rows["reason_target"].shape[1]
    if thresholds is not None and thresholds.numel() != action_count + reason_count:
        raise ValueError("deployment threshold count must match action+reason labels")
    return {
        "action": fit_object_intent_deployment_gates(
            rows["pre_object_intent_action"], rows["object_intent_action_candidate"],
            rows["action_target"], min_samples=min_samples,
            min_nll_improvement=min_nll_improvement,
            thresholds=None if thresholds is None else thresholds[:action_count],
            min_f1_improvement=min_f1_improvement,
        ),
        "reason": fit_object_intent_deployment_gates(
            rows["pre_object_intent_reason"], rows["object_intent_reason_candidate"],
            rows["reason_target"], min_samples=min_samples,
            min_nll_improvement=min_nll_improvement,
            thresholds=None if thresholds is None else thresholds[action_count:],
            min_f1_improvement=min_f1_improvement,
        ),
    }


def apply_object_intent_gates_to_rows(
    rows: dict[str, torch.Tensor], action_gate: torch.Tensor, reason_gate: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rows = dict(rows)
    action_delta = rows["object_intent_action_candidate"] * action_gate[None]
    reason_delta = rows["object_intent_reason_candidate"] * reason_gate[None]
    rows["object_intent_action_delta"] = action_delta
    rows["object_intent_reason_delta"] = reason_delta
    for key in (
        "object_intent_action_selected_deleted_delta",
        "object_intent_action_control_deleted_delta",
    ):
        rows[key] = rows[key] * action_gate[None]
    for key in (
        "object_intent_reason_selected_deleted_delta",
        "object_intent_reason_control_deleted_delta",
    ):
        rows[key] = rows[key] * reason_gate[None]
    for branch, gate in (("action", action_gate), ("reason", reason_gate)):
        pair_candidate = f"object_intent_{branch}_pair_candidate"
        if pair_candidate in rows:
            rows[f"object_intent_{branch}_pair_delta"] = rows[pair_candidate] * gate[None]
            for kind in ("selected", "control"):
                source = f"object_intent_{branch}_{kind}_pair_deleted_candidate"
                rows[f"object_intent_{branch}_{kind}_pair_deleted_delta"] = (
                    rows[source] * gate[None]
                )
    rows["video_action"] = rows["pre_object_intent_action"] + action_delta
    rows["video_reason"] = rows["pre_object_intent_reason"] + reason_delta
    return rows


def apply_object_intent_utility_policy_to_rows(
    rows: dict[str, torch.Tensor],
    action_policy: dict[str, torch.Tensor],
    reason_policy: dict[str, torch.Tensor],
    *,
    action_cap: float = 0.08,
    reason_cap: float = 0.06,
) -> dict[str, torch.Tensor]:
    rows = dict(rows)

    def apply(branch: str, policy: dict[str, torch.Tensor], cap: float) -> torch.Tensor:
        candidate = rows[f"object_intent_{branch}_candidate"]
        utility = rows[f"object_intent_{branch}_utility_gate"]
        gate = policy["gate"].to(candidate)
        scale = policy["scale"].to(candidate)
        cutoff = policy["cutoff"].to(candidate)
        selected = (gate[None] > 0) & (utility >= cutoff[None])
        delta = gate[None] * selected.to(candidate.dtype) * (
            candidate * scale[None]
        ).clamp(-float(cap), float(cap))
        rows[f"object_intent_{branch}_delta"] = delta
        rows[f"object_intent_{branch}_deploy_gate"] = gate[None].expand_as(candidate)
        rows[f"object_intent_{branch}_deploy_scale"] = scale[None].expand_as(candidate)
        rows[f"object_intent_{branch}_utility_cutoff"] = cutoff[None].expand_as(candidate)
        rows[f"object_intent_{branch}_utility_selected"] = selected.to(candidate.dtype)
        for kind in ("selected", "control"):
            key = f"object_intent_{branch}_{kind}_deleted_candidate"
            if key in rows:
                rows[f"object_intent_{branch}_{kind}_deleted_delta"] = (
                    gate[None] * selected.to(candidate.dtype)
                    * (rows[key] * scale[None]).clamp(-float(cap), float(cap))
                )
            pair_key = f"object_intent_{branch}_{kind}_pair_deleted_candidate"
            if pair_key in rows:
                rows[f"object_intent_{branch}_{kind}_pair_deleted_delta"] = (
                    gate[None] * selected.to(candidate.dtype)
                    * (rows[pair_key] * scale[None]).clamp(-float(cap), float(cap))
                )
        if f"object_intent_{branch}_pair_candidate" in rows:
            rows[f"object_intent_{branch}_pair_delta"] = (
                gate[None] * selected.to(candidate.dtype)
                * (rows[f"object_intent_{branch}_pair_candidate"] * scale[None]).clamp(
                    -float(cap), float(cap)
                )
            )
        return delta

    action_delta = apply("action", action_policy, action_cap)
    reason_delta = apply("reason", reason_policy, reason_cap)
    rows["video_action"] = rows["pre_object_intent_action"] + action_delta
    rows["video_reason"] = rows["pre_object_intent_reason"] + reason_delta
    return rows


def _augment_target_effectiveness(
    result: dict[str, object],
    *,
    base: torch.Tensor,
    final: torch.Tensor,
    target: torch.Tensor,
    delta: torch.Tensor,
    selected_deleted: torch.Tensor,
    control_deleted: torch.Tensor,
    thresholds: torch.Tensor,
) -> None:
    target_bool = target > 0.5
    base_prediction = base.sigmoid() >= thresholds[None]
    final_prediction = final.sigmoid() >= thresholds[None]
    base_correct = base_prediction == target_bool
    final_correct = final_prediction == target_bool
    recovered = (~base_correct) & final_correct
    damaged = base_correct & (~final_correct)
    sign = 2.0 * target.float() - 1.0
    signed_gain = sign * (final - base)
    selected_damage = sign * (delta - selected_deleted)
    control_damage = sign * (delta - control_deleted)
    target_effective = (
        (signed_gain > 0)
        & (selected_damage > 0)
        & (selected_damage > control_damage)
    )
    labels = max(target.numel(), 1)
    base_errors = (~base_correct).sum().clamp_min(1)
    result.update(
        {
            "target_effective_route_rate": float(target_effective.float().mean()),
            "target_effective_route_rate_by_label": target_effective.float().mean(0).tolist(),
            "errors_recovered": int(recovered.sum()),
            "correct_predictions_damaged": int(damaged.sum()),
            "net_corrected_labels": int(recovered.sum() - damaged.sum()),
            "net_corrected_per_1000_labels": float(
                1000.0 * (recovered.sum() - damaged.sum()).float() / labels
            ),
            "temporal_error_recovery_rate": float(recovered.sum().float() / base_errors),
            "temporal_damage_rate": float(damaged.float().mean()),
        }
    )


def _augment_decoupled_route_metrics(
    result: dict[str, object],
    *,
    semantic_attention: torch.Tensor,
    motion_attention: torch.Tensor,
    motion_mix: torch.Tensor,
    interaction_risk: torch.Tensor,
) -> None:
    eps = 1e-8
    semantic = semantic_attention.clamp_min(eps)
    motion = motion_attention.clamp_min(eps)
    midpoint = 0.5 * (semantic + motion)
    jsd = 0.5 * (
        (semantic * (semantic.log() - midpoint.log())).sum(-1)
        + (motion * (motion.log() - midpoint.log())).sum(-1)
    )
    semantic_risk = torch.einsum("nlk,nk->nl", semantic_attention, interaction_risk)
    motion_risk = torch.einsum("nlk,nk->nl", motion_attention, interaction_risk)
    disagreement = semantic_attention.argmax(-1) != motion_attention.argmax(-1)
    result.update(
        {
            "motion_semantic_attention_jsd": float(jsd.mean()),
            "motion_semantic_attention_jsd_by_label": jsd.mean(0).tolist(),
            "motion_semantic_selected_track_disagreement_rate": float(
                disagreement.float().mean()
            ),
            "motion_semantic_selected_track_disagreement_rate_by_label": (
                disagreement.float().mean(0).tolist()
            ),
            "motion_mix_mean": float(motion_mix.mean()),
            "motion_mix_p50": float(motion_mix.quantile(0.5)),
            "motion_mix_p95": float(motion_mix.quantile(0.95)),
            "motion_mix_by_label": motion_mix.mean(0).tolist(),
            "semantic_attention_risk_mean": float(semantic_risk.mean()),
            "motion_attention_risk_mean": float(motion_risk.mean()),
            "motion_minus_semantic_risk_focus": float(
                (motion_risk - semantic_risk).mean()
            ),
            "motion_minus_semantic_risk_focus_by_label": (
                motion_risk - semantic_risk
            ).mean(0).tolist(),
        }
    )


def _pair_interaction_effectiveness(
    rows: dict[str, torch.Tensor], branch: str,
) -> dict[str, object] | None:
    pair_key = f"object_intent_{branch}_pair_attention"
    if pair_key not in rows:
        return None
    attention = rows[pair_key].float()
    flat_attention = attention.flatten(2)
    entropy = -(
        flat_attention * flat_attention.clamp_min(1e-8).log()
    ).sum(-1)
    selected = flat_attention.argmax(-1)
    tracks = attention.shape[-1]
    selected_min_distance = torch.gather(
        rows["object_intent_pair_min_future_distance"].reshape(-1, tracks * tracks)[:, None].expand(
            -1, selected.shape[1], -1
        ), 2, selected[..., None]
    ).squeeze(-1)
    selected_reduction = torch.gather(
        rows["object_intent_pair_distance_reduction"].reshape(-1, tracks * tracks)[:, None].expand(
            -1, selected.shape[1], -1
        ), 2, selected[..., None]
    ).squeeze(-1)
    target = rows[f"{branch}_target"].float()
    sign = 2.0 * target - 1.0
    pair_delta_key = f"object_intent_{branch}_pair_delta"
    selected_delta_key = f"object_intent_{branch}_selected_pair_deleted_delta"
    control_delta_key = f"object_intent_{branch}_control_pair_deleted_delta"
    if pair_delta_key in rows:
        pair_delta = rows[pair_delta_key].float()
        selected_deleted = rows[selected_delta_key].float()
        control_deleted = rows[control_delta_key].float()
    else:
        gate = rows[f"object_intent_{branch}_deploy_gate"].float()
        pair_delta = rows[f"object_intent_{branch}_pair_candidate"].float() * gate
        selected_deleted = (
            rows[f"object_intent_{branch}_selected_pair_deleted_candidate"].float() * gate
        )
        control_deleted = (
            rows[f"object_intent_{branch}_control_pair_deleted_candidate"].float() * gate
        )
    full_delta = rows[f"object_intent_{branch}_delta"].float()
    selected_damage = sign * (full_delta - selected_deleted)
    control_damage = sign * (full_delta - control_deleted)
    nonzero = pair_delta.abs() > 1e-8
    precision = ((sign * pair_delta) > 0)[nonzero].float().mean() if nonzero.any() else pair_delta.new_tensor(0.0)
    return {
        "pair_attention_entropy_mean": float(entropy.mean()),
        "pair_attention_max_mean": float(flat_attention.amax(-1).mean()),
        "selected_pair_min_future_distance_mean": float(selected_min_distance.mean()),
        "selected_pair_distance_reduction_mean": float(selected_reduction.mean()),
        "pair_delta_rms": float(pair_delta.square().mean().sqrt()),
        "pair_transport_precision": float(precision),
        "selected_pair_deletion_damage": float(selected_damage.mean()),
        "control_pair_deletion_damage": float(control_damage.mean()),
        "selected_minus_control_pair_deletion_gap": float(
            (selected_damage - control_damage).mean()
        ),
        "pair_route_coverage": float((flat_attention.amax(-1) > 0).float().mean()),
    }


def _utility_quality(
    rows: dict[str, torch.Tensor], branch: str,
) -> dict[str, object] | None:
    key = f"object_intent_{branch}_utility_gate"
    if key not in rows:
        return None
    utility = rows[key].float()
    candidate = rows[f"object_intent_{branch}_candidate"].float()
    target = rows[f"{branch}_target"].float()
    helpful = ((2.0 * target - 1.0) * candidate) > 0
    selected = rows.get(
        f"object_intent_{branch}_utility_selected", utility >= 0.5
    ).bool()
    selected_count = selected.sum().clamp_min(1)
    positive_count = int(helpful.sum())
    negative_count = int((~helpful).sum())
    if positive_count and negative_count:
        flat_scores = utility.flatten()
        flat_helpful = helpful.flatten()
        order = flat_scores.argsort()
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(
            1, order.numel() + 1, device=order.device, dtype=torch.float32
        )
        rank_sum = ranks[flat_helpful].sum()
        auc = float(
            (rank_sum - positive_count * (positive_count + 1) / 2)
            / (positive_count * negative_count)
        )
    else:
        auc = None
    flat_utility = utility.flatten()
    flat_margin = ((2.0 * target - 1.0) * candidate).flatten()
    order = flat_utility.argsort(descending=True)
    curve = []
    for coverage in (0.25, 0.50, 0.75, 1.0):
        count = max(1, int(round(order.numel() * coverage)))
        chosen = order[:count]
        curve.append({
            "coverage": coverage,
            "benefit_rate": float((flat_margin[chosen] > 0).float().mean()),
            "harm_rate": float((flat_margin[chosen] < 0).float().mean()),
            "signed_margin_mean": float(flat_margin[chosen].mean()),
        })
    scale = rows.get(f"object_intent_{branch}_deploy_scale")
    return {
        "helpfulness_auc": auc,
        "selected_rate": float(selected.float().mean()),
        "selected_benefit_rate": float((helpful & selected).sum() / selected_count),
        "selected_harm_rate": float(((~helpful) & selected).sum() / selected_count),
        "utility_mean": float(utility.mean()),
        "utility_p50": float(utility.quantile(0.50)),
        "utility_p95": float(utility.quantile(0.95)),
        "deploy_scale_mean": None if scale is None else float(scale.float().mean()),
        "deploy_scale_max_abs": None if scale is None else float(scale.float().abs().max()),
        "coverage_risk_curve": curve,
    }


def object_intent_traffic_metrics(
    rows: dict[str, torch.Tensor],
    thresholds: torch.Tensor,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 20260825,
) -> dict[str, object]:
    action_count = rows["action_target"].shape[1]
    reason_count = rows["reason_target"].shape[1]
    if thresholds.numel() != action_count + reason_count:
        raise ValueError("threshold count must match action and reason labels")
    mapped = {
        "pre_relational_action": rows["pre_object_intent_action"],
        "pre_relational_reason": rows["pre_object_intent_reason"],
        "video_action": rows["video_action"],
        "video_reason": rows["video_reason"],
        "action_target": rows["action_target"].float(),
        "reason_target": rows["reason_target"].float(),
        "relational_action_delta": rows["object_intent_action_delta"],
        "relational_reason_delta": rows["object_intent_reason_delta"],
        "relational_action_selected_deleted_delta": rows[
            "object_intent_action_selected_deleted_delta"
        ],
        "relational_action_random_deleted_delta": rows[
            "object_intent_action_control_deleted_delta"
        ],
        "relational_reason_selected_deleted_delta": rows[
            "object_intent_reason_selected_deleted_delta"
        ],
        "relational_reason_random_deleted_delta": rows[
            "object_intent_reason_control_deleted_delta"
        ],
        "relational_action_support": rows["object_intent_action_support"],
        "relational_reason_support": rows["object_intent_reason_support"],
        "relational_action_attention": rows["object_intent_action_attention"],
        "relational_reason_attention": rows["object_intent_reason_attention"],
        "relational_interaction_risk": rows["object_intent_interaction_risk"][..., None],
    }
    result = relational_traffic_metrics(
        mapped, bootstrap_samples=bootstrap_samples, seed=seed
    )
    _augment_target_effectiveness(
        result["action"],
        base=mapped["pre_relational_action"],
        final=mapped["video_action"],
        target=mapped["action_target"],
        delta=mapped["relational_action_delta"],
        selected_deleted=mapped["relational_action_selected_deleted_delta"],
        control_deleted=mapped["relational_action_random_deleted_delta"],
        thresholds=thresholds[:action_count],
    )
    _augment_target_effectiveness(
        result["reason"],
        base=mapped["pre_relational_reason"],
        final=mapped["video_reason"],
        target=mapped["reason_target"],
        delta=mapped["relational_reason_delta"],
        selected_deleted=mapped["relational_reason_selected_deleted_delta"],
        control_deleted=mapped["relational_reason_random_deleted_delta"],
        thresholds=thresholds[action_count:],
    )
    for branch in ("action", "reason"):
        pair_metrics = _pair_interaction_effectiveness(rows, branch)
        if pair_metrics is not None:
            result[branch]["future_pair_interaction"] = pair_metrics
        utility_metrics = _utility_quality(rows, branch)
        if utility_metrics is not None:
            result[branch]["utility_quality"] = utility_metrics
    risk = rows["object_intent_interaction_risk"]
    _augment_decoupled_route_metrics(
        result["action"],
        semantic_attention=rows["object_intent_action_semantic_attention"],
        motion_attention=rows["object_intent_action_motion_attention"],
        motion_mix=rows["object_intent_action_motion_mix"],
        interaction_risk=risk,
    )
    _augment_decoupled_route_metrics(
        result["reason"],
        semantic_attention=rows["object_intent_reason_semantic_attention"],
        motion_attention=rows["object_intent_reason_motion_attention"],
        motion_mix=rows["object_intent_reason_motion_mix"],
        interaction_risk=risk,
    )
    if "object_intent_track_role_probs" in rows:
        role_availability = rows["object_intent_track_role_probs"].mean((0, 1))
        for branch in ("action", "reason"):
            role_mass = rows[f"object_intent_{branch}_role_mass"]
            mean_mass = role_mass.mean((0, 1))
            lift = mean_mass - role_availability
            result[branch].update({
                "role_names": ROLE_NAMES,
                "predicted_track_role_availability": role_availability.tolist(),
                "target_attention_role_mass": mean_mass.tolist(),
                "target_attention_role_lift": lift.tolist(),
                "foreground_attention_mass": float(role_mass[..., 1:].sum(-1).mean()),
                "role_selectivity_std": float(role_mass.mean(0).std(0).mean()),
                "role_mass_by_label": role_mass.mean(0).tolist(),
            })
    if "object_intent_track_role_consistency" in rows:
        consistency = rows["object_intent_track_role_consistency"]
        weights = rows["object_intent_semantic_temporal_weights"].clamp_min(0)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1)
        result["temporal_identity"] = {
            "role_consistency_mean": float(consistency.mean()),
            "role_consistency_p10": float(consistency.quantile(0.10)),
            "role_consistency_p50": float(consistency.quantile(0.50)),
            "role_consistency_p90": float(consistency.quantile(0.90)),
            "effective_visible_frames_mean": float(entropy.exp().mean()),
            "visible_frame_count_mean": float((weights > 0).float().sum(1).mean()),
            "temporal_weight_entropy_mean": float(entropy.mean()),
        }
    result["traffic_complexity_strata"] = _traffic_complexity_strata(rows, thresholds)
    if "object_intent_future_approach_risk" in rows:
        result["future_approach_effectiveness"] = _future_approach_effectiveness(
            rows, thresholds
        )
    return result
