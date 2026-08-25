from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _bootstrap_mean(value: torch.Tensor, samples: int, seed: int) -> list[float]:
    value = value.detach().float().flatten().cpu()
    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randint(value.numel(), (int(samples), value.numel()), generator=generator)
    means = value[index].mean(-1)
    return [float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))]


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2 or left.numel() != right.numel():
        return float("nan")
    left_rank = left.argsort().argsort().float()
    right_rank = right.argsort().argsort().float()
    left_rank = left_rank - left_rank.mean()
    right_rank = right_rank - right_rank.mean()
    denominator = left_rank.square().sum().sqrt() * right_rank.square().sum().sqrt()
    return float((left_rank * right_rank).sum() / denominator.clamp_min(1e-8))


def _target_metrics(
    base: torch.Tensor,
    final: torch.Tensor,
    target: torch.Tensor,
    delta: torch.Tensor,
    selected_deleted: torch.Tensor,
    random_deleted: torch.Tensor,
    support: torch.Tensor,
    attention: torch.Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    target = target.float()
    sign = 2.0 * target - 1.0
    signed_margin = sign * (final - base)
    base_nll = F.binary_cross_entropy_with_logits(base, target, reduction="none").mean(-1)
    final_nll = F.binary_cross_entropy_with_logits(final, target, reduction="none").mean(-1)
    nll_gain = base_nll - final_nll
    base_brier = (base.sigmoid() - target).square().mean(-1)
    final_brier = (final.sigmoid() - target).square().mean(-1)
    brier_gain = base_brier - final_brier
    selected_damage = sign * (delta - selected_deleted)
    random_damage = sign * (delta - random_deleted)
    deletion_gap = (selected_damage - random_damage).mean(-1)
    necessity = (selected_damage > random_damage) & (selected_damage > 0)
    entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
    return {
        "conditional_nll_improvement": float(nll_gain.mean()),
        # Proper-score information gain states how many target-label bits the
        # transported traffic evidence adds beyond the visual baseline.
        "conditional_information_gain_bits": float(nll_gain.mean() / math.log(2.0)),
        "conditional_nll_improvement_ci95": _bootstrap_mean(
            nll_gain, bootstrap_samples, seed
        ),
        "conditional_brier_improvement": float(brier_gain.mean()),
        "relative_brier_reduction": float(
            brier_gain.mean() / base_brier.mean().clamp_min(1e-8)
        ),
        "conditional_brier_improvement_ci95": _bootstrap_mean(
            brier_gain, bootstrap_samples, seed + 1
        ),
        "signed_margin_mean": float(signed_margin.mean()),
        "signed_margin_benefit_rate": float((signed_margin > 0).float().mean()),
        "signed_margin_harm_rate": float((signed_margin < 0).float().mean()),
        "signed_margin_by_label": signed_margin.mean(0).tolist(),
        "selected_deletion_damage": float(selected_damage.mean()),
        "random_deletion_damage": float(random_damage.mean()),
        "selected_minus_random_deletion_gap": float(deletion_gap.mean()),
        "selected_minus_random_deletion_gap_ci95": _bootstrap_mean(
            deletion_gap, bootstrap_samples, seed + 2
        ),
        "route_necessity_precision": float(necessity.float().mean()),
        "transport_support_mean": float(support.mean()),
        "transport_coverage_rate": float((support > 0.02).float().mean()),
        "target_attention_entropy_mean": float(entropy.mean()),
        "target_attention_max_mean": float(attention.max(-1).values.mean()),
        "delta_rms": float(delta.square().mean().sqrt()),
    }


def relational_traffic_metrics(
    rows: dict[str, torch.Tensor],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 20260825,
) -> dict[str, object]:
    action = _target_metrics(
        rows["pre_relational_action"], rows["video_action"], rows["action_target"],
        rows["relational_action_delta"],
        rows["relational_action_selected_deleted_delta"],
        rows["relational_action_random_deleted_delta"],
        rows["relational_action_support"], rows["relational_action_attention"],
        bootstrap_samples=bootstrap_samples, seed=seed,
    )
    reason = _target_metrics(
        rows["pre_relational_reason"], rows["video_reason"], rows["reason_target"],
        rows["relational_reason_delta"],
        rows["relational_reason_selected_deleted_delta"],
        rows["relational_reason_random_deleted_delta"],
        rows["relational_reason_support"], rows["relational_reason_attention"],
        bootstrap_samples=bootstrap_samples, seed=seed + 10,
    )
    risk = rows["relational_interaction_risk"].amax(dim=(-1, -2))
    action_sample_gain = (
        F.binary_cross_entropy_with_logits(
            rows["pre_relational_action"], rows["action_target"].float(), reduction="none"
        )
        - F.binary_cross_entropy_with_logits(
            rows["video_action"], rows["action_target"].float(), reduction="none"
        )
    ).mean(-1)
    reason_sample_gain = (
        F.binary_cross_entropy_with_logits(
            rows["pre_relational_reason"], rows["reason_target"].float(), reduction="none"
        )
        - F.binary_cross_entropy_with_logits(
            rows["video_reason"], rows["reason_target"].float(), reduction="none"
        )
    ).mean(-1)
    if risk.numel() < 4:
        raise ValueError("risk-stratified traffic metrics require at least four clips")
    risk_order = risk.argsort()
    quartile_rows: list[dict[str, object]] = []
    for index in range(4):
        begin = index * risk.numel() // 4
        end = (index + 1) * risk.numel() // 4
        selected = risk_order[begin:end]
        mask = torch.zeros_like(risk, dtype=torch.bool)
        mask[selected] = True
        action_sign = 2.0 * rows["action_target"][mask].float() - 1.0
        reason_sign = 2.0 * rows["reason_target"][mask].float() - 1.0
        action_gap = action_sign * (
            rows["relational_action_random_deleted_delta"][mask]
            - rows["relational_action_selected_deleted_delta"][mask]
        )
        reason_gap = reason_sign * (
            rows["relational_reason_random_deleted_delta"][mask]
            - rows["relational_reason_selected_deleted_delta"][mask]
        )
        quartile_rows.append({
            "quartile": index + 1,
            "count": int(mask.sum()),
            "risk_min": float(risk[mask].min()),
            "risk_max": float(risk[mask].max()),
            "action_information_gain_bits": float(
                action_sample_gain[mask].mean() / math.log(2.0)
            ),
            "reason_information_gain_bits": float(
                reason_sample_gain[mask].mean() / math.log(2.0)
            ),
            "action_deletion_gap": float(action_gap.mean()),
            "reason_deletion_gap": float(reason_gap.mean()),
        })
    threshold = torch.quantile(risk, 0.75)
    high = risk >= threshold
    if high.any():
        action_target = rows["action_target"][high]
        reason_target = rows["reason_target"][high]
        action_base_nll = F.binary_cross_entropy_with_logits(
            rows["pre_relational_action"][high], action_target, reduction="none"
        ).mean(-1)
        action_final_nll = F.binary_cross_entropy_with_logits(
            rows["video_action"][high], action_target, reduction="none"
        ).mean(-1)
        reason_base_nll = F.binary_cross_entropy_with_logits(
            rows["pre_relational_reason"][high], reason_target, reduction="none"
        ).mean(-1)
        reason_final_nll = F.binary_cross_entropy_with_logits(
            rows["video_reason"][high], reason_target, reduction="none"
        ).mean(-1)
        action_sign = 2.0 * action_target - 1.0
        reason_sign = 2.0 * reason_target - 1.0
        action_deletion_gap = (
            action_sign * (
                rows["relational_action_random_deleted_delta"][high]
                - rows["relational_action_selected_deleted_delta"][high]
            )
        ).mean()
        reason_deletion_gap = (
            reason_sign * (
                rows["relational_reason_random_deleted_delta"][high]
                - rows["relational_reason_selected_deleted_delta"][high]
            )
        ).mean()
        high_result: dict[str, object] = {
            "available": True,
            "count": int(high.sum()),
            "risk_threshold": float(threshold),
            "action_signed_margin_mean": float(
                ((2.0 * action_target - 1.0) * (
                    rows["video_action"][high] - rows["pre_relational_action"][high]
                )).mean()
            ),
            "reason_signed_margin_mean": float(
                ((2.0 * reason_target - 1.0) * (
                    rows["video_reason"][high] - rows["pre_relational_reason"][high]
                )).mean()
            ),
            "action_information_gain_bits": float(
                (action_base_nll - action_final_nll).mean() / math.log(2.0)
            ),
            "reason_information_gain_bits": float(
                (reason_base_nll - reason_final_nll).mean() / math.log(2.0)
            ),
            "action_deletion_gap": float(action_deletion_gap),
            "reason_deletion_gap": float(reason_deletion_gap),
        }
    else:
        high_result = {"available": False, "count": 0}
    risk_association = {
        "action_spearman": _spearman(risk, action_sample_gain),
        "reason_spearman": _spearman(risk, reason_sample_gain),
        "high_minus_low_action_information_gain_bits": (
            quartile_rows[-1]["action_information_gain_bits"]
            - quartile_rows[0]["action_information_gain_bits"]
        ),
        "high_minus_low_reason_information_gain_bits": (
            quartile_rows[-1]["reason_information_gain_bits"]
            - quartile_rows[0]["reason_information_gain_bits"]
        ),
    }
    return {
        "action": action,
        "reason": reason,
        "high_interaction_risk": high_result,
        "interaction_risk_quartiles": quartile_rows,
        "risk_utility_association": risk_association,
    }
