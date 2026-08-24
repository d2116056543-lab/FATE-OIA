from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.tida_contracts import _best_label_threshold
from fate_oia.utils.tida_artifacts import atomic_write_json
from fate_oia.utils.tida_temporal_metrics import paired_temporal_contribution, robust_motion_score


def gt_margin_advantage(
    real_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    sign = 2.0 * target.to(real_logits.dtype) - 1.0
    return sign * (real_logits - counterfactual_logits)


def _binary_rank_auc(score: torch.Tensor, target: torch.Tensor) -> float | None:
    score = score.flatten().float()
    target = target.flatten().bool()
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return None
    order = score.argsort()
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, score.numel() + 1, device=score.device, dtype=score.dtype)
    numerator = ranks[target].sum() - positives * (positives + 1) / 2.0
    return float(numerator / (positives * negatives))


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def select_predicate_intervention_indices(
    route: torch.Tensor,
    contribution: torch.Tensor,
    *,
    count: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if route.shape != contribution.shape or route.ndim != 3:
        raise ValueError("route and contribution must both be [B,A,F]")
    if int(count) * 2 > route.shape[-1]:
        raise ValueError("factor bank is too small for disjoint selected/control sets")
    route_score = route.mean((0, 1))
    contribution_score = contribution.abs().mean((0, 1))
    selected = contribution_score.topk(int(count)).indices
    available = torch.ones_like(route_score, dtype=torch.bool)
    available[selected] = False
    controls = []
    for selected_index in selected:
        distance = (route_score - route_score[selected_index]).abs().masked_fill(~available, float("inf"))
        control = distance.argmin()
        controls.append(control)
        available[control] = False
    return selected, torch.stack(controls)


@torch.no_grad()
def collect_tida_outputs(
    model, loader, device: torch.device, *, temporal_scale: float = 1.0,
    collect_mechanism: bool = False, mechanism_samples: int = 128,
    collect_audit_tensors: bool = False,
) -> dict[str, Any]:
    store = {key: [] for key in (
        "image_action", "semantic_action", "geometric_action", "traffic_action", "trajectory_action",
        "semantic_trajectory_action", "video_action_base", "video_action",
        "image_reason", "semantic_reason", "geometric_reason", "video_reason",
        "prefix_action", "prefix_reason", "action_target", "reason_target",
    )}
    diagnostics = {key: [] for key in (
        "rho", "action_delta", "reason_delta", "null_mass", "route_entropy",
        "action_evidence_confidence", "action_effective_trust",
        "reason_evidence_confidence", "reason_effective_trust",
        "action_flow_route_mass", "reason_flow_route_mass", "transition_reliability",
        "action_temporal_budget", "reason_temporal_budget",
        "action_temporal_need", "reason_temporal_need",
        "action_temporal_target_motion", "reason_temporal_target_motion",
        "reason_pu_weight",
        "velocity_norm", "acceleration_norm",
        "geometric_motion_energy", "geometric_global_horizontal", "geometric_global_expansion",
        "geometric_region_motion", "geometric_action_delta", "geometric_reason_delta",
        "traffic_motion_energy", "traffic_action_delta", "traffic_action_attention",
        "traffic_same_action_mass",
        "traffic_patch_displacement", "traffic_patch_common_displacement",
        "traffic_patch_exclusive_displacement", "traffic_patch_match_confidence",
        "traffic_patch_motion_energy", "traffic_patch_exclusive_motion_energy",
        "traffic_patch_effective_motion",
        "traffic_trajectory_delta", "traffic_trajectory_control_delta",
        "traffic_trajectory_candidate_delta", "traffic_trajectory_utility_logit",
        "traffic_trajectory_utility_gate",
        "traffic_trajectory_order_utility_gate",
        "traffic_trajectory_state_utility_logit", "traffic_trajectory_state_utility_gate",
        "traffic_adaptive_boundary_delta", "traffic_adaptive_deploy_action_logits",
        "traffic_trajectory_order_delta", "traffic_trajectory_state_delta",
        "traffic_trajectory_state_effective_delta", "traffic_trajectory_state_logit",
        "traffic_trajectory_state_features", "trajectory_state_strength",
        "traffic_trajectory_support", "trajectory_support_gate",
        "trajectory_order_gate", "trajectory_uncertainty_gate",
        "trajectory_attention",
        "trajectory_speed", "trajectory_acceleration", "trajectory_radial_motion",
        "trajectory_order_contrast_rms",
        "trajectory_cycle_confidence", "trajectory_common_displacement",
        "trajectory_exclusive_displacement", "trajectory_xy",
        "trajectory_local_candidate_coverage", "trajectory_interaction_risk",
    )}
    audit_keys = (
        "terminal_prediction_history", "terminal_prediction_no_history", "terminal_target_evidence",
        "terminal_error_history", "terminal_error_no_history", "innovation_token",
        "predicate_differential_state", "predicate_velocity_norm", "predicate_acceleration_norm",
        "predicate_persistence", "predicate_region_mass", "predicate_region_mass_velocity", "common_motion_norm",
        "transition_tokens", "transition_tokens_by_scale", "motion_salience", "transition_consistency",
        "velocity", "acceleration", "region_velocity", "transition_reliability",
        "action_flow_route_mass", "reason_flow_route_mass",
        "action_temporal_route", "action_factor_contribution", "reason_temporal_route", "frame_valid_mask", "timestamps",
    )
    audit_store: dict[str, list[torch.Tensor]] = {key: [] for key in audit_keys}
    dynamic_concepts: list[dict[str, Any]] = []
    file_names: list[str] = []
    mechanism_rows: dict[str, list[float]] = {
        name: [] for name in (
            "history_off", "repeated_last", "time_shuffle", "time_reverse",
            "selected_predicate_flatten", "matched_predicate_flatten", "wrong_action_route",
            "static_only", "dynamic_only",
        )
    }
    mechanism_outputs: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {key: [] for key in ("action", "reason", "velocity")} for name in mechanism_rows
    }
    mechanism_base = {key: [] for key in ("action", "reason", "action_target", "reason_target", "velocity")}
    mechanism_count = 0
    model.eval()
    for batch in loader:
        batch = _device_batch(batch, device)
        output = model(
            batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"],
            temporal_action_scale=temporal_scale, temporal_reason_scale=temporal_scale,
        )
        values = {
            "image_action": output["image_action_logits"],
            "semantic_action": output["semantic_video_action_logits"],
            "geometric_action": output["geometric_video_action_logits"],
            "traffic_action": output["traffic_video_action_logits"],
            "trajectory_action": output["trajectory_video_action_logits"],
            "semantic_trajectory_action": output["semantic_trajectory_video_action_logits"],
            "video_action_base": output["video_action_logits_base"],
            "video_action": output["video_action_logits"],
            "image_reason": output["image_reason_logits"],
            "semantic_reason": output["semantic_video_reason_logits"],
            "geometric_reason": output["geometric_video_reason_logits"],
            "video_reason": output["video_reason_logits"],
            "prefix_action": output["prefix_video_action_logits"],
            "prefix_reason": output["prefix_video_reason_logits"],
            "action_target": batch["action"], "reason_target": batch["reason"],
        }
        for key, value in values.items():
            store[key].append(value.detach().float().cpu())
        image_branch = output.get("image_branch", {})
        contradiction = image_branch.get("contradiction_score") if isinstance(image_branch, dict) else None
        reason_negative_weight = (
            torch.full_like(batch["reason"], 0.2)
            if contradiction is None
            else 0.2 + 0.8 * contradiction.detach().clamp(0.0, 1.0)
        )
        reason_pu = torch.where(batch["reason"] > 0.5, torch.ones_like(reason_negative_weight), reason_negative_weight)
        for key, value in {
            "rho": output["innovation_reliability"], "action_delta": output["action_temporal_delta"],
            "reason_delta": output["reason_temporal_delta"], "null_mass": output["action_null_mass"],
            "route_entropy": output["action_route_entropy"],
            "action_evidence_confidence": output["action_evidence_confidence"],
            "action_effective_trust": output["action_effective_trust"],
            "reason_evidence_confidence": output["reason_evidence_confidence"],
            "reason_effective_trust": output["reason_effective_trust"],
            "action_flow_route_mass": output["action_flow_route_mass"],
            "reason_flow_route_mass": output["reason_flow_route_mass"],
            "transition_reliability": output["transition_reliability"],
            "action_temporal_budget": output["action_temporal_budget"],
            "reason_temporal_budget": output["reason_temporal_budget"],
            "action_temporal_need": output["action_temporal_need"],
            "reason_temporal_need": output["reason_temporal_need"],
            "action_temporal_target_motion": output["action_temporal_target_motion"],
            "reason_temporal_target_motion": output["reason_temporal_target_motion"],
            "reason_pu_weight": reason_pu,
            "velocity_norm": output["velocity"].norm(dim=-1),
            "acceleration_norm": output["acceleration"].norm(dim=-1),
            "geometric_motion_energy": output["geometric_motion_energy"],
            "geometric_global_horizontal": output["geometric_global_horizontal"],
            "geometric_global_expansion": output["geometric_global_expansion"],
            "geometric_region_motion": output["geometric_region_motion"],
            "geometric_action_delta": output["geometric_action_delta"],
            "geometric_reason_delta": output["geometric_reason_delta_effective"],
            "traffic_motion_energy": output["traffic_motion_energy"],
            "traffic_action_delta": output["traffic_action_delta"],
            "traffic_action_attention": output["traffic_action_attention"],
            "traffic_same_action_mass": output["traffic_same_action_mass"],
            "traffic_patch_displacement": output["traffic_patch_displacement"],
            "traffic_patch_common_displacement": output["traffic_patch_common_displacement"],
            "traffic_patch_exclusive_displacement": output["traffic_patch_exclusive_displacement"],
            "traffic_patch_match_confidence": output["traffic_patch_match_confidence"],
            "traffic_patch_motion_energy": output["traffic_patch_motion_energy"],
            "traffic_patch_exclusive_motion_energy": output["traffic_patch_exclusive_motion_energy"],
            "traffic_patch_effective_motion": output["traffic_patch_effective_motion"],
            "traffic_trajectory_delta": output["traffic_trajectory_delta"],
            "traffic_trajectory_control_delta": output["traffic_trajectory_control_delta"],
            "traffic_trajectory_candidate_delta": output["traffic_trajectory_candidate_delta"],
            "traffic_trajectory_utility_logit": output["traffic_trajectory_utility_logit"],
            "traffic_trajectory_utility_gate": output["traffic_trajectory_utility_gate"],
            "traffic_trajectory_order_utility_gate": output["traffic_trajectory_order_utility_gate"],
            "traffic_trajectory_state_utility_logit": output["traffic_trajectory_state_utility_logit"],
            "traffic_trajectory_state_utility_gate": output["traffic_trajectory_state_utility_gate"],
            "traffic_adaptive_boundary_delta": output["traffic_adaptive_boundary_delta"],
            "traffic_adaptive_deploy_action_logits": output["traffic_adaptive_deploy_action_logits"],
            "traffic_trajectory_order_delta": output["traffic_trajectory_order_delta"],
            "traffic_trajectory_state_delta": output["traffic_trajectory_state_delta"],
            "traffic_trajectory_state_effective_delta": output[
                "traffic_trajectory_state_effective_delta"
            ],
            "traffic_trajectory_state_logit": output["traffic_trajectory_state_logit"],
            "traffic_trajectory_state_features": output["traffic_trajectory_state_features"],
            "trajectory_state_strength": output["trajectory_state_strength"],
            "traffic_trajectory_support": output["traffic_trajectory_support"],
            "trajectory_support_gate": output["trajectory_support_gate"],
            "trajectory_order_gate": output["trajectory_order_gate"],
            "trajectory_uncertainty_gate": output["trajectory_uncertainty_gate"],
            "trajectory_attention": output["trajectory_attention"],
            "trajectory_speed": output["trajectory_speed"],
            "trajectory_acceleration": output["trajectory_acceleration"],
            "trajectory_radial_motion": output["trajectory_radial_motion"],
            "trajectory_order_contrast_rms": output["trajectory_order_contrast_rms"],
            "trajectory_cycle_confidence": output["trajectory_cycle_confidence"],
            "trajectory_common_displacement": output["trajectory_common_displacement"],
            "trajectory_exclusive_displacement": output["trajectory_exclusive_displacement"],
            "trajectory_xy": output["trajectory_xy"],
            "trajectory_local_candidate_coverage": output[
                "trajectory_local_candidate_coverage"
            ],
            "trajectory_interaction_risk": output["trajectory_interaction_risk"],
        }.items():
            diagnostics[key].append(value.detach().float().cpu())
        if collect_audit_tensors:
            for key in audit_keys:
                value = output[key].detach().cpu()
                audit_store[key].append(value if key == "frame_valid_mask" else value.float())
            dynamic_concepts.extend(output["dynamic_concepts"])
        file_names.extend(batch["file_name"])
        if collect_mechanism and mechanism_count < mechanism_samples:
            selected, matched = select_predicate_intervention_indices(
                output["action_route"][..., :32],
                output["action_factor_contribution"][..., :32],
            )
            mechanism_base["action"].append(output["video_action_logits"].detach().float().cpu())
            mechanism_base["reason"].append(output["video_reason_logits"].detach().float().cpu())
            mechanism_base["action_target"].append(batch["action"].detach().float().cpu())
            mechanism_base["reason_target"].append(batch["reason"].detach().float().cpu())
            mechanism_base["velocity"].append(output["velocity"].detach().float().cpu())
            for name in mechanism_rows:
                if "predicate_flatten" in name:
                    indices = selected if name.startswith("selected") else matched
                    output["intervention_predicate_indices"] = tuple(int(value) for value in indices)
                changed = model.rerun_temporal_from_output(
                    output, name, temporal_action_scale=temporal_scale, temporal_reason_scale=temporal_scale
                )
                mechanism_rows[name].append(float(
                    (changed["terminal_error_history"].mean() - output["terminal_error_history"].mean()).cpu()
                ))
                mechanism_outputs[name]["action"].append(changed["video_action_logits"].detach().float().cpu())
                mechanism_outputs[name]["reason"].append(changed["video_reason_logits"].detach().float().cpu())
                mechanism_outputs[name]["velocity"].append(changed["velocity"].detach().float().cpu())
            mechanism_count += batch["target_image"].shape[0]
    missing_collections = [
        key for key, values in (store | diagnostics).items() if not values
    ]
    if missing_collections:
        raise RuntimeError(
            "collect_tida_outputs did not collect declared fields: "
            + ", ".join(missing_collections)
        )
    result = {key: torch.cat(value) for key, value in store.items()} | {
        key: torch.cat(value) for key, value in diagnostics.items()
    } | {"file_names": file_names}
    if collect_audit_tensors:
        result.update({key: torch.cat(value) for key, value in audit_store.items()})
        result["dynamic_concepts"] = dynamic_concepts
    if collect_mechanism:
        base_rows = {
            "image_action": torch.cat(mechanism_base["action"]), "video_action": torch.cat(mechanism_base["action"]),
            "image_reason": torch.cat(mechanism_base["reason"]), "video_reason": torch.cat(mechanism_base["reason"]),
            "action_target": torch.cat(mechanism_base["action_target"]), "reason_target": torch.cat(mechanism_base["reason_target"]),
        }
        base_metric = branch_metrics(base_rows)["video"]
        intervention_metrics = {}
        for name, values in mechanism_outputs.items():
            changed_rows = dict(base_rows)
            changed_rows["video_action"] = torch.cat(values["action"])
            changed_rows["video_reason"] = torch.cat(values["reason"])
            metric = branch_metrics(changed_rows)["video"]
            changed_action = torch.cat(values["action"])
            changed_reason = torch.cat(values["reason"])
            action_advantage = gt_margin_advantage(
                base_rows["video_action"], changed_action, base_rows["action_target"]
            )
            reason_advantage = gt_margin_advantage(
                base_rows["video_reason"], changed_reason, base_rows["reason_target"]
            )
            changed_velocity = torch.cat(values["velocity"])
            real_velocity = torch.cat(mechanism_base["velocity"])
            velocity_cosine = torch.nn.functional.cosine_similarity(
                real_velocity.flatten(1), changed_velocity.flatten(1), dim=-1
            )
            intervention_metrics[name] = {
                **metric,
                "joint_drop_from_real": base_metric["joint"] - metric["joint"],
                "action_mf1_drop_from_real": base_metric["Act_mF1"] - metric["Act_mF1"],
                "reason_mf1_drop_from_real": base_metric["Exp_mF1"] - metric["Exp_mF1"],
                "action_gt_margin_advantage_mean": float(action_advantage.mean()),
                "reason_gt_margin_advantage_mean": float(reason_advantage.mean()),
                "action_gt_margin_advantage_by_label": action_advantage.mean(0).tolist(),
                "reason_gt_margin_advantage_by_label": reason_advantage.mean(0).tolist(),
                "velocity_cosine_with_reference": float(velocity_cosine.mean()),
            }
        result["_mechanism"] = {
            "available": mechanism_count > 0,
            "sample_count": min(mechanism_count, mechanism_samples),
            "target_dino_reruns": 0,
            "mean_error_increase": {
                name: sum(values) / max(len(values), 1) for name, values in mechanism_rows.items()
            },
            "real_history_metrics": base_metric,
            "intervention_metrics": intervention_metrics,
        }
    return result


def branch_metrics(rows: dict[str, Any], thresholds: torch.Tensor | float = 0.5) -> dict[str, Any]:
    return {
        "image": aie_branch_metrics(rows["image_action"], rows["image_reason"], rows["action_target"], rows["reason_target"], threshold=thresholds),
        "video": aie_branch_metrics(rows["video_action"], rows["video_reason"], rows["action_target"], rows["reason_target"], threshold=thresholds),
    }


def dynamic_slice_metrics(rows: dict[str, Any], thresholds: torch.Tensor | float = 0.5) -> dict[str, Any]:
    score = rows["rho"].mean(-1)
    masks = {"low_dynamic": score <= 0.10, "high_dynamic": score >= 0.25}
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            result[name] = {"available": False, "count": 0}
            continue
        sliced = {key: value[mask] for key, value in rows.items() if torch.is_tensor(value) and value.shape[0] == mask.shape[0]}
        metrics = branch_metrics(sliced, thresholds)
        result[name] = {
            "available": True, "count": int(mask.sum()), "rho_mean": float(score[mask].mean()),
            "image": metrics["image"], "video": metrics["video"],
            "action_mf1_delta": metrics["video"]["Act_mF1"] - metrics["image"]["Act_mF1"],
            "reason_mf1_delta": metrics["video"]["Exp_mF1"] - metrics["image"]["Exp_mF1"],
        }
    return result


def temporal_contribution_metrics(rows: dict[str, Any]) -> dict[str, Any]:
    motion = robust_motion_score(rows["velocity_norm"], rows["acceleration_norm"])
    return {
        "action": paired_temporal_contribution(
            rows["image_action"], rows["video_action"], rows["action_target"], motion_score=motion,
        ),
        "reason": paired_temporal_contribution(
            rows["image_reason"], rows["video_reason"], rows["reason_target"], motion_score=motion,
            pu_negative_weight=rows.get("reason_pu_weight"),
        ),
    }


def geometric_branch_metrics(rows: dict[str, Any], thresholds: torch.Tensor | float = 0.5) -> dict[str, Any]:
    branches = {
        "image_only": ("image_action", "image_reason"),
        "semantic_temporal_only": ("semantic_action", "semantic_reason"),
        "geometric_temporal_only": ("geometric_action", "geometric_reason"),
        "semantic_plus_geometric": ("video_action", "video_reason"),
    }
    return {
        name: aie_branch_metrics(
            rows[action_key], rows[reason_key], rows["action_target"], rows["reason_target"], threshold=thresholds
        )
        for name, (action_key, reason_key) in branches.items()
    }


def geometric_temporal_effectiveness_metrics(
    rows: dict[str, Any], thresholds: torch.Tensor | float = 0.5
) -> dict[str, Any]:
    score = rows["geometric_motion_energy"].mean(1)
    low_cut = torch.quantile(score, 0.25)
    high_cut = torch.quantile(score, 0.75)
    subset_rows = {}
    for name, mask in (("low_motion", score <= low_cut), ("high_motion", score >= high_cut)):
        sliced = {
            key: value[mask] for key, value in rows.items()
            if torch.is_tensor(value) and value.shape[0] == mask.shape[0]
        }
        branch = geometric_branch_metrics(sliced, thresholds)
        semantic = branch["semantic_temporal_only"]
        final = branch["semantic_plus_geometric"]
        subset_rows[name] = {
            "count": int(mask.sum()), "motion_energy_mean": float(score[mask].mean()),
            "branches": branch,
            "geometric_incremental_action_mf1": final["Act_mF1"] - semantic["Act_mF1"],
            "geometric_incremental_reason_mf1": final["Exp_mF1"] - semantic["Exp_mF1"],
            "geometric_incremental_action_map": final["Act_mAP"] - semantic["Act_mAP"],
            "geometric_incremental_reason_map": final["Exp_mAP"] - semantic["Exp_mAP"],
        }

    prefix_metrics = []
    for index, fraction in enumerate((0.25, 0.50, 0.75, 1.0)):
        metric = aie_branch_metrics(
            rows["prefix_action"][:, index], rows["prefix_reason"][:, index],
            rows["action_target"], rows["reason_target"], threshold=thresholds,
        )
        prefix_metrics.append({"history_fraction": fraction, **metric})
    action_auc = sum(row["Act_mF1"] for row in prefix_metrics) / len(prefix_metrics)
    reason_auc = sum(row["Exp_mF1"] for row in prefix_metrics) / len(prefix_metrics)

    action_sign = 2.0 * rows["action_target"] - 1.0
    reason_sign = 2.0 * rows["reason_target"] - 1.0
    action_margin = action_sign * rows["geometric_action_delta"]
    reason_margin = reason_sign * rows["geometric_reason_delta"]
    return {
        "motion_quantiles": {
            "p25": float(low_cut), "p50": float(torch.quantile(score, 0.5)), "p75": float(high_cut)
        },
        "subsets": subset_rows,
        "anticipation_curve": prefix_metrics,
        "anticipation_auc": {"action_mf1": action_auc, "reason_mf1": reason_auc},
        "target_transport": {
            "action_signed_margin_mean": float(action_margin.mean()),
            "reason_signed_margin_mean": float(reason_margin.mean()),
            "action_benefit_rate": float((action_margin > 0).float().mean()),
            "reason_benefit_rate": float((reason_margin > 0).float().mean()),
            "action_signed_margin_by_label": action_margin.mean(0).tolist(),
            "reason_signed_margin_by_label": reason_margin.mean(0).tolist(),
            "action_delta_rms": float(rows["geometric_action_delta"].square().mean().sqrt()),
            "reason_delta_rms": float(rows["geometric_reason_delta"].square().mean().sqrt()),
        },
    }


def traffic_action_effectiveness_metrics(
    rows: dict[str, Any], thresholds: torch.Tensor | float = 0.5
) -> dict[str, Any]:
    """Measure whether ordered traffic motion improves a specific action target."""
    semantic = aie_branch_metrics(
        rows["semantic_action"], rows["semantic_reason"],
        rows["action_target"], rows["reason_target"], threshold=thresholds,
    )
    final = aie_branch_metrics(
        rows["video_action"], rows["video_reason"],
        rows["action_target"], rows["reason_target"], threshold=thresholds,
    )
    traffic_only = aie_branch_metrics(
        rows["traffic_action"], rows["image_reason"],
        rows["action_target"], rows["reason_target"], threshold=thresholds,
    )
    if "traffic_patch_effective_motion" in rows:
        score = rows["traffic_patch_effective_motion"].mean((1, 2))
        motion_source = "exclusive_patch_motion_x_match_confidence"
    else:
        score = rows["traffic_motion_energy"].mean(1)
        motion_source = "action_token_velocity"
    low_cut, high_cut = torch.quantile(score, 0.25), torch.quantile(score, 0.75)
    strata: dict[str, Any] = {}
    for name, mask in (("low_motion", score <= low_cut), ("high_motion", score >= high_cut)):
        semantic_slice = aie_branch_metrics(
            rows["semantic_action"][mask], rows["semantic_reason"][mask],
            rows["action_target"][mask], rows["reason_target"][mask], threshold=thresholds,
        )
        final_slice = aie_branch_metrics(
            rows["video_action"][mask], rows["video_reason"][mask],
            rows["action_target"][mask], rows["reason_target"][mask], threshold=thresholds,
        )
        strata[name] = {
            "count": int(mask.sum()),
            "motion_energy_mean": float(score[mask].mean()),
            "traffic_incremental_action_mf1": final_slice["Act_mF1"] - semantic_slice["Act_mF1"],
            "traffic_incremental_action_map": final_slice["Act_mAP"] - semantic_slice["Act_mAP"],
            "semantic": semantic_slice,
            "final": final_slice,
            }

    benefit_curve = []
    boundaries = torch.quantile(score, torch.linspace(0, 1, 6, device=score.device))
    for index in range(5):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (score >= lower) & ((score <= upper) if index == 4 else (score < upper))
        if not mask.any():
            benefit_curve.append({"bin": index, "count": 0, "available": False})
            continue
        semantic_slice = aie_branch_metrics(
            rows["semantic_action"][mask], rows["semantic_reason"][mask],
            rows["action_target"][mask], rows["reason_target"][mask], threshold=thresholds,
        )
        final_slice = aie_branch_metrics(
            rows["video_action"][mask], rows["video_reason"][mask],
            rows["action_target"][mask], rows["reason_target"][mask], threshold=thresholds,
        )
        benefit_curve.append({
            "bin": index, "count": int(mask.sum()), "available": True,
            "motion_mean": float(score[mask].mean()),
            "action_mf1_delta": final_slice["Act_mF1"] - semantic_slice["Act_mF1"],
            "action_map_delta": final_slice["Act_mAP"] - semantic_slice["Act_mAP"],
        })

    threshold = torch.as_tensor(thresholds, device=score.device, dtype=rows["semantic_action"].dtype)
    if threshold.ndim:
        threshold = threshold.flatten()[: rows["action_target"].shape[1]]
    semantic_pred = torch.sigmoid(rows["semantic_action"]) >= threshold
    final_pred = torch.sigmoid(rows["video_action"]) >= threshold
    positive = rows["action_target"] > 0.5
    flip_counts = {
        "fn_to_tp": ((~semantic_pred) & final_pred & positive).sum(0).tolist(),
        "fp_to_tn": (semantic_pred & (~final_pred) & (~positive)).sum(0).tolist(),
        "tp_to_fn": (semantic_pred & (~final_pred) & positive).sum(0).tolist(),
        "tn_to_fp": ((~semantic_pred) & final_pred & (~positive)).sum(0).tolist(),
    }
    sign = 2.0 * rows["action_target"] - 1.0
    signed_margin = sign * rows["traffic_action_delta"]
    attention = rows["traffic_action_attention"]
    attention_entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
    normalizer = torch.log(torch.tensor(max(attention.shape[-1], 2), dtype=attention.dtype))
    batch, actions = rows["action_target"].shape
    intervals = rows["traffic_motion_energy"].shape[1]
    patch_available = "traffic_patch_displacement" in rows
    patch_displacement = rows.get(
        "traffic_patch_displacement", attention.new_zeros(batch, intervals, actions, 2)
    )
    patch_common = rows.get(
        "traffic_patch_common_displacement", attention.new_zeros(batch, intervals, 2)
    )
    patch_exclusive = rows.get(
        "traffic_patch_exclusive_displacement", attention.new_zeros(batch, intervals, actions, 2)
    )
    patch_confidence = rows.get(
        "traffic_patch_match_confidence", attention.new_zeros(batch, intervals, actions)
    )
    patch_energy = rows.get(
        "traffic_patch_motion_energy", attention.new_zeros(batch, intervals, actions)
    )
    return {
        "overall": {
            "semantic": semantic,
            "traffic_only": traffic_only,
            "final": final,
            "traffic_incremental_action_mf1": final["Act_mF1"] - semantic["Act_mF1"],
            "traffic_incremental_action_of1": final["Act_oF1"] - semantic["Act_oF1"],
            "traffic_incremental_action_map": final["Act_mAP"] - semantic["Act_mAP"],
        },
        "motion_quantiles": {
            "source": motion_source,
            "p25": float(low_cut), "p50": float(torch.quantile(score, 0.5)), "p75": float(high_cut),
        },
        "motion_strata": strata,
        "dynamic_benefit_curve": benefit_curve,
        "corrective_flip_counts_by_action": flip_counts,
        "target_transport": {
            "action_signed_margin_mean": float(signed_margin.mean()),
            "action_benefit_rate": float((signed_margin > 0).float().mean()),
            "action_signed_margin_by_label": signed_margin.mean(0).tolist(),
            "action_benefit_rate_by_label": (signed_margin > 0).float().mean(0).tolist(),
            "action_delta_rms": float(rows["traffic_action_delta"].square().mean().sqrt()),
        },
        "attention": {
            "normalized_entropy_mean": float((attention_entropy / normalizer).mean()),
            "same_action_mass_mean": float(rows["traffic_same_action_mass"].mean()),
            "same_action_mass_by_target": rows["traffic_same_action_mass"].mean(0).tolist(),
            "patch_correspondence_available": patch_available,
            "patch_match_confidence_mean": float(patch_confidence.mean()),
            "patch_motion_energy_mean": float(patch_energy.mean()),
            "patch_displacement_xy_mean": patch_displacement.mean((0, 1)).tolist(),
            "patch_common_displacement_xy_mean": patch_common.mean((0, 1)).tolist(),
            "patch_exclusive_displacement_xy_by_action": patch_exclusive.mean((0, 1)).tolist(),
            "patch_exclusive_motion_rms_by_action": patch_exclusive.square().mean((0, 1, 3)).sqrt().tolist(),
        },
    }


def trajectory_traffic_effectiveness_metrics(
    rows: dict[str, Any], thresholds: torch.Tensor | float = 0.5
) -> dict[str, Any]:
    """Quantify target transport, dynamic utility, grounding quality, and causal dependence."""
    semantic = aie_branch_metrics(
        rows["semantic_action"], rows["image_reason"], rows["action_target"], rows["reason_target"],
        threshold=thresholds,
    )
    trajectory = aie_branch_metrics(
        rows["semantic_trajectory_action"], rows["image_reason"],
        rows["action_target"], rows["reason_target"],
        threshold=thresholds,
    )
    final = aie_branch_metrics(
        rows["video_action"], rows["video_reason"], rows["action_target"], rows["reason_target"],
        threshold=thresholds,
    )
    score = rows["trajectory_speed"].mean((1, 2, 3))
    low_cut, high_cut = torch.quantile(score, 0.25), torch.quantile(score, 0.75)
    dynamic = {}
    for name, mask in (("low_motion", score <= low_cut), ("high_motion", score >= high_cut)):
        base_slice = aie_branch_metrics(
            rows["semantic_action"][mask], rows["image_reason"][mask],
            rows["action_target"][mask], rows["reason_target"][mask], threshold=thresholds,
        )
        trajectory_slice = aie_branch_metrics(
            rows["semantic_trajectory_action"][mask], rows["image_reason"][mask],
            rows["action_target"][mask], rows["reason_target"][mask], threshold=thresholds,
        )
        dynamic[name] = {
            "count": int(mask.sum()), "motion_mean": float(score[mask].mean()),
            "Act_mF1_semantic": base_slice["Act_mF1"],
            "Act_mF1_trajectory": trajectory_slice["Act_mF1"],
            "Act_mF1_gain": trajectory_slice["Act_mF1"] - base_slice["Act_mF1"],
            "Act_mAP_gain": trajectory_slice["Act_mAP"] - base_slice["Act_mAP"],
        }
    sign = 2.0 * rows["action_target"] - 1.0
    signed = sign * rows["traffic_trajectory_delta"]
    signed_control_advantage = sign * (
        rows["traffic_trajectory_delta"] - rows["traffic_trajectory_control_delta"]
    )
    state_effective_delta = rows.get(
        "traffic_trajectory_state_effective_delta", torch.zeros_like(rows["traffic_trajectory_delta"])
    )
    state_order_delta = rows.get(
        "traffic_trajectory_order_delta", rows["traffic_trajectory_delta"]
    )
    state_strength = rows.get(
        "trajectory_state_strength", torch.zeros_like(rows["traffic_trajectory_delta"])
    )
    signed_state = sign * state_effective_delta
    benefit = int((signed > 1e-4).sum())
    harm = int((signed < -1e-4).sum())
    attention = rows["trajectory_attention"]
    entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
    normalized_entropy = entropy / torch.log(torch.tensor(max(attention.shape[-1], 2), dtype=entropy.dtype))
    paired = paired_temporal_contribution(
        rows["semantic_action"], rows["semantic_trajectory_action"], rows["action_target"],
        motion_score=score, bootstrap_samples=500,
    )
    mechanism = rows.get("_mechanism", {})
    interventions = mechanism.get("intervention_metrics", {}) if isinstance(mechanism, dict) else {}
    causal = {
        name: {
            "action_mf1_drop_from_ordered": value.get("action_mf1_drop_from_real"),
            "action_gt_margin_advantage_mean": value.get("action_gt_margin_advantage_mean"),
        }
        for name, value in interventions.items()
        if name in {"time_shuffle", "time_reverse", "repeated_last", "history_off"}
    }


    threshold = torch.as_tensor(
        thresholds, device=rows["semantic_action"].device, dtype=rows["semantic_action"].dtype
    )
    if threshold.ndim:
        threshold = threshold.flatten()[: rows["action_target"].shape[1]]
    semantic_pred = torch.sigmoid(rows["semantic_action"]) >= threshold
    trajectory_pred = torch.sigmoid(rows["semantic_trajectory_action"]) >= threshold
    positive = rows["action_target"] > 0.5
    decision_flips = {
        "fn_to_tp": ((~semantic_pred) & trajectory_pred & positive).sum(0).tolist(),
        "fp_to_tn": (semantic_pred & (~trajectory_pred) & (~positive)).sum(0).tolist(),
        "tp_to_fn": (semantic_pred & (~trajectory_pred) & positive).sum(0).tolist(),
        "tn_to_fp": ((~semantic_pred) & trajectory_pred & (~positive)).sum(0).tolist(),
    }
    local_coverage = rows.get("trajectory_local_candidate_coverage")
    common_displacement = rows.get("trajectory_common_displacement")
    interaction_risk = rows.get("trajectory_interaction_risk")
    utility_gate = rows.get("traffic_trajectory_utility_gate")
    candidate_delta = rows.get("traffic_trajectory_order_delta")
    utility_quality = {"available": False}
    if utility_gate is not None and candidate_delta is not None:
        candidate_helpful = sign * candidate_delta > 0
        selected = utility_gate >= 0.5
        selected_signed = signed[selected]
        utility_quality = {
            "available": True,
            "gate_mean": float(utility_gate.mean()),
            "gate_p50": float(torch.quantile(utility_gate, 0.50)),
            "gate_p95": float(torch.quantile(utility_gate, 0.95)),
            "selected_rate": float(selected.float().mean()),
            "helpfulness_auc": _binary_rank_auc(utility_gate, candidate_helpful),
            "selected_benefit_rate": (
                0.0 if selected_signed.numel() == 0 else float((selected_signed > 1e-4).float().mean())
            ),
            "selected_harm_rate": (
                0.0 if selected_signed.numel() == 0 else float((selected_signed < -1e-4).float().mean())
            ),
        }
    state_utility_gate = rows.get("traffic_trajectory_state_utility_gate")
    state_utility_quality = {"available": False}
    if state_utility_gate is not None:
        state_candidate_helpful = sign * rows["traffic_trajectory_state_delta"] > 0
        state_selected = state_utility_gate >= 0.5
        state_selected_signed = signed_state[state_selected]
        state_utility_quality = {
            "available": True,
            "gate_mean": float(state_utility_gate.mean()),
            "gate_p50": float(torch.quantile(state_utility_gate, 0.50)),
            "gate_p95": float(torch.quantile(state_utility_gate, 0.95)),
            "selected_rate": float(state_selected.float().mean()),
            "helpfulness_auc": _binary_rank_auc(state_utility_gate, state_candidate_helpful),
            "selected_benefit_rate": (
                0.0 if state_selected_signed.numel() == 0
                else float((state_selected_signed > 1e-4).float().mean())
            ),
            "selected_harm_rate": (
                0.0 if state_selected_signed.numel() == 0
                else float((state_selected_signed < -1e-4).float().mean())
            ),
        }
    state_risk_correlation = None
    if interaction_risk is not None:
        state_flat = state_strength.flatten().float()
        risk_flat = interaction_risk.mean(-1).flatten().float()
        if state_flat.std() > 1e-8 and risk_flat.std() > 1e-8:
            state_risk_correlation = float(torch.corrcoef(torch.stack((state_flat, risk_flat)))[0, 1])
    return {
        "overall": {
            "semantic": semantic, "trajectory_only": trajectory, "final": final,
            "trajectory_incremental_action_mf1": trajectory["Act_mF1"] - semantic["Act_mF1"],
            "trajectory_incremental_action_map": trajectory["Act_mAP"] - semantic["Act_mAP"],
        },
        "dynamic_conditioned": dynamic,
        "target_transport": {
            "action_signed_margin_mean": float(signed.mean()),
            "action_signed_margin_by_label": signed.mean(0).tolist(),
            "ordered_control_advantage_mean": float(signed_control_advantage.mean()),
            "ordered_control_advantage_by_label": signed_control_advantage.mean(0).tolist(),
            "benefit_count": benefit, "harm_count": harm,
            "benefit_rate": float((signed > 1e-4).float().mean()),
            "harm_rate": float((signed < -1e-4).float().mean()),
            "correction_to_harm_ratio": float((benefit + 1) / (harm + 1)),
            "paired_bootstrap": paired,
        },
        "motion_state_transport": {
            "signed_margin_mean": float(signed_state.mean()),
            "signed_margin_by_label": signed_state.mean(0).tolist(),
            "benefit_rate": float((signed_state > 1e-4).float().mean()),
            "harm_rate": float((signed_state < -1e-4).float().mean()),
            "state_delta_rms": float(
                state_effective_delta.square().mean().sqrt()
            ),
            "order_delta_rms": float(state_order_delta.square().mean().sqrt()),
            "state_strength_mean": float(state_strength.mean()),
            "interaction_risk_correlation": state_risk_correlation,
            "utility_quality": state_utility_quality,
        },
        "grounding_quality": {
            "support_mean": float(rows["traffic_trajectory_support"].mean()),
            "support_gate_mean": float(rows["trajectory_support_gate"].mean()),
            "supported_action_rate": float((rows["traffic_trajectory_support"] > 0.20).float().mean()),
            "cycle_confidence_mean": float(rows["trajectory_cycle_confidence"].mean()),
            "normalized_attention_entropy_mean": float(normalized_entropy.mean()),
            "effective_track_count_mean": float(torch.exp(entropy).mean()),
            "exclusive_motion_rms": float(rows["trajectory_exclusive_displacement"].square().mean().sqrt()),
            "order_contrast_rms_mean": float(rows["trajectory_order_contrast_rms"].mean()),
            "order_gate_mean": float(rows["trajectory_order_gate"].mean()),
            "uncertainty_gate_mean": float(rows["trajectory_uncertainty_gate"].mean()),
            "interaction_risk_mean": (
                None if interaction_risk is None else float(interaction_risk.mean())
            ),
            "dense_local_matching_available": local_coverage is not None,
            "local_candidate_coverage_mean": (
                None if local_coverage is None else float(local_coverage.mean())
            ),
            "exclusive_to_common_motion_ratio": (
                None
                if common_displacement is None
                else float(
                    rows["trajectory_exclusive_displacement"].square().mean().sqrt()
                    / common_displacement.square().mean().sqrt().clamp_min(1e-8)
                )
            ),
        },
        "utility_quality": utility_quality,
        "decision_flips": decision_flips,
        "causal_temporal_interventions": causal,
    }


def traffic_adaptive_boundary_effectiveness_metrics(
    rows: dict[str, Any], thresholds: torch.Tensor | float = 0.5
) -> dict[str, Any]:
    """Attribute deploy changes to the traffic-conditioned decision boundary."""
    base = aie_branch_metrics(
        rows["video_action_base"], rows["video_reason"],
        rows["action_target"], rows["reason_target"], threshold=thresholds,
    )
    adaptive = aie_branch_metrics(
        rows["video_action"], rows["video_reason"],
        rows["action_target"], rows["reason_target"], threshold=thresholds,
    )
    delta = rows["traffic_adaptive_boundary_delta"]
    # Deploy logits are base - boundary_delta, so -delta is the GT-margin change.
    signed_margin = (2.0 * rows["action_target"] - 1.0) * (-delta)
    threshold = torch.as_tensor(
        thresholds, device=delta.device, dtype=delta.dtype
    ).flatten()
    action_threshold = threshold[: delta.shape[1]] if threshold.numel() > 1 else threshold
    base_pred = torch.sigmoid(rows["video_action_base"]) >= action_threshold
    adaptive_pred = torch.sigmoid(rows["video_action"]) >= action_threshold
    positive = rows["action_target"] > 0.5
    motion = rows["trajectory_speed"].mean((1, 2, 3))
    risk = rows["trajectory_interaction_risk"].mean((1, 2))
    conditioned: dict[str, Any] = {}
    for score_name, score in (("motion", motion), ("interaction_risk", risk)):
        low, high = torch.quantile(score, 0.25), torch.quantile(score, 0.75)
        for band, mask in (("low", score <= low), ("high", score >= high)):
            base_slice = aie_branch_metrics(
                rows["video_action_base"][mask], rows["video_reason"][mask],
                rows["action_target"][mask], rows["reason_target"][mask],
                threshold=thresholds,
            )
            adaptive_slice = aie_branch_metrics(
                rows["video_action"][mask], rows["video_reason"][mask],
                rows["action_target"][mask], rows["reason_target"][mask],
                threshold=thresholds,
            )
            conditioned[f"{band}_{score_name}"] = {
                "count": int(mask.sum()),
                "score_mean": float(score[mask].mean()),
                "Act_mF1_base": base_slice["Act_mF1"],
                "Act_mF1_adaptive": adaptive_slice["Act_mF1"],
                "Act_mF1_gain": adaptive_slice["Act_mF1"] - base_slice["Act_mF1"],
                "Act_mAP_gain": adaptive_slice["Act_mAP"] - base_slice["Act_mAP"],
            }
    return {
        "overall": {
            "base": base,
            "adaptive": adaptive,
            "Act_mF1_gain": adaptive["Act_mF1"] - base["Act_mF1"],
            "Act_oF1_gain": adaptive["Act_oF1"] - base["Act_oF1"],
            "Act_mAP_gain": adaptive["Act_mAP"] - base["Act_mAP"],
        },
        "transport": {
            "delta_mean_by_action": delta.mean(0).tolist(),
            "delta_rms_by_action": delta.square().mean(0).sqrt().tolist(),
            "delta_rms": float(delta.square().mean().sqrt()),
            "gt_margin_mean": float(signed_margin.mean()),
            "gt_margin_by_action": signed_margin.mean(0).tolist(),
            "gt_margin_sign_agreement": float((signed_margin > 0.0).float().mean()),
        },
        "decision_flips": {
            "fn_to_tp": ((~base_pred) & adaptive_pred & positive).sum(0).tolist(),
            "fp_to_tn": (base_pred & (~adaptive_pred) & (~positive)).sum(0).tolist(),
            "tp_to_fn": (base_pred & (~adaptive_pred) & positive).sum(0).tolist(),
            "tn_to_fp": ((~base_pred) & adaptive_pred & (~positive)).sum(0).tolist(),
        },
        "dynamic_conditioned": conditioned,
        "reason_firewall": {"reason_logit_delta_rms": 0.0},
    }


def fit_train_calib_thresholds(rows: dict[str, Any]) -> dict[str, torch.Tensor]:
    video_logits = torch.cat([rows["video_action"], rows["video_reason"]], dim=-1)
    image_logits = torch.cat([rows["image_action"], rows["image_reason"]], dim=-1)
    targets = torch.cat([rows["action_target"], rows["reason_target"]], dim=-1)
    return {
        "video": _best_label_threshold(video_logits, targets),
        "image": _best_label_threshold(image_logits, targets),
    }


@torch.no_grad()
def collect_intervention_audit(model, loader, device: torch.device, max_samples: int = 128) -> dict[str, Any]:
    interventions = (
        "history_off", "repeated_last", "time_shuffle", "time_reverse",
        "wrong_action_route", "static_only", "dynamic_only",
    )
    rows: dict[str, list[float]] = {name: [] for name in interventions}
    count = 0
    target_encode_calls = 0
    for batch in loader:
        batch = _device_batch(batch, device)
        output = model(
            batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"],
            temporal_action_scale=1.0, temporal_reason_scale=1.0,
        )
        target_encode_calls += 1
        base = output["terminal_error_history"].mean()
        for name in interventions:
            changed = model.rerun_temporal_from_output(
                output, name, temporal_action_scale=1.0, temporal_reason_scale=1.0
            )
            rows[name].append(float((changed["terminal_error_history"].mean() - base).cpu()))
        count += batch["target_image"].shape[0]
        if count >= max_samples:
            break
    return {
        "available": count > 0,
        "sample_count": min(count, max_samples),
        "target_encode_calls": target_encode_calls,
        "mean_error_increase": {name: sum(values) / max(len(values), 1) for name, values in rows.items()},
    }


def save_epoch_outputs(output_dir: Path, epoch: int, rows: dict[str, Any], metrics: dict[str, Any], thresholds: dict[str, torch.Tensor], mechanism: dict[str, Any]) -> None:
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(epoch_dir / "metrics_summary.json", metrics)
    atomic_write_json(epoch_dir / "calibration.json", {key: value.tolist() for key, value in thresholds.items()})
    atomic_write_json(epoch_dir / "temporal_mechanism_audit.json", mechanism)
    contribution = metrics.get("online", {}).get("temporal_contribution")
    if contribution is not None:
        atomic_write_json(epoch_dir / "temporal_contribution_metrics.json", contribution)
    geometric_effectiveness = metrics.get("online", {}).get("geometric_effectiveness")
    if geometric_effectiveness is not None:
        atomic_write_json(epoch_dir / "geometric_temporal_effectiveness.json", geometric_effectiveness)
    traffic_effectiveness = metrics.get("online", {}).get("traffic_action_effectiveness")
    if traffic_effectiveness is not None:
        atomic_write_json(epoch_dir / "traffic_action_effectiveness.json", traffic_effectiveness)
    trajectory_effectiveness = metrics.get("online", {}).get("trajectory_traffic_effectiveness")
    if trajectory_effectiveness is not None:
        atomic_write_json(
            epoch_dir / "trajectory_traffic_effectiveness.json", trajectory_effectiveness
        )
    boundary_effectiveness = metrics.get("online", {}).get(
        "traffic_adaptive_boundary_effectiveness"
    )
    if boundary_effectiveness is not None:
        atomic_write_json(
            epoch_dir / "traffic_adaptive_boundary_effectiveness.json", boundary_effectiveness
        )
    atomic_write_json(epoch_dir / "file_names_test.json", rows["file_names"])
    if "dynamic_concepts" in rows:
        with (epoch_dir / "dynamic_concepts_test.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for file_name, concepts in zip(rows["file_names"], rows["dynamic_concepts"]):
                handle.write(json.dumps({"file_name": file_name, "dynamic_concepts": concepts}, ensure_ascii=False) + "\n")
    tensor_keys = (
        "image_action", "semantic_action", "geometric_action", "traffic_action",
        "video_action_base", "video_action",
        "image_reason", "semantic_reason", "geometric_reason", "video_reason",
        "prefix_action", "prefix_reason", "action_target", "reason_target",
        "rho", "action_delta", "reason_delta", "null_mass", "route_entropy",
        "action_evidence_confidence", "action_effective_trust",
        "reason_evidence_confidence", "reason_effective_trust",
        "action_flow_route_mass", "reason_flow_route_mass", "transition_reliability",
        "action_temporal_budget", "reason_temporal_budget",
        "action_temporal_need", "reason_temporal_need",
        "action_temporal_target_motion", "reason_temporal_target_motion",
        "reason_pu_weight",
        "velocity_norm", "acceleration_norm",
        "geometric_motion_energy", "geometric_global_horizontal", "geometric_global_expansion",
        "geometric_region_motion", "geometric_action_delta", "geometric_reason_delta",
        "traffic_motion_energy", "traffic_action_delta", "traffic_action_attention",
        "traffic_same_action_mass",
        "traffic_patch_displacement", "traffic_patch_common_displacement",
        "traffic_patch_exclusive_displacement", "traffic_patch_match_confidence",
        "traffic_patch_motion_energy", "traffic_patch_exclusive_motion_energy",
        "traffic_patch_effective_motion",
        "traffic_trajectory_delta", "traffic_trajectory_control_delta",
        "traffic_trajectory_candidate_delta", "traffic_trajectory_utility_logit",
        "traffic_trajectory_utility_gate",
        "traffic_trajectory_order_utility_gate",
        "traffic_trajectory_state_utility_logit", "traffic_trajectory_state_utility_gate",
        "traffic_adaptive_boundary_delta", "traffic_adaptive_deploy_action_logits",
        "traffic_trajectory_order_delta", "traffic_trajectory_state_delta",
        "traffic_trajectory_state_effective_delta", "traffic_trajectory_state_logit",
        "traffic_trajectory_state_features", "trajectory_state_strength",
        "traffic_trajectory_support", "trajectory_support_gate",
        "trajectory_order_gate", "trajectory_uncertainty_gate",
        "trajectory_attention",
        "trajectory_speed", "trajectory_acceleration", "trajectory_radial_motion",
        "trajectory_order_contrast_rms",
        "trajectory_cycle_confidence", "trajectory_common_displacement",
        "trajectory_exclusive_displacement", "trajectory_xy",
        "trajectory_local_candidate_coverage", "trajectory_interaction_risk",
        "terminal_prediction_history", "terminal_prediction_no_history", "terminal_target_evidence",
        "terminal_error_history", "terminal_error_no_history", "innovation_token",
        "predicate_differential_state", "predicate_velocity_norm", "predicate_acceleration_norm",
        "predicate_persistence", "predicate_region_mass", "predicate_region_mass_velocity", "common_motion_norm",
        "transition_tokens", "transition_tokens_by_scale", "motion_salience", "transition_consistency",
        "velocity", "acceleration", "region_velocity",
        "action_temporal_route", "action_factor_contribution", "reason_temporal_route", "frame_valid_mask", "timestamps",
    )
    for key in tensor_keys:
        if key not in rows:
            continue
        torch.save(rows[key], epoch_dir / f"{key}_test.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-view", choices=("online", "ema"), default="online")
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--collect-mechanism", action="store_true")
    args = parser.parse_args()
    from fate_oia.engine.train_tida_oia import build_runtime

    runtime = build_runtime(args, evaluation_only=True)
    rows = collect_tida_outputs(
        runtime.model,
        runtime.loaders["test"],
        runtime.device,
        collect_mechanism=args.collect_mechanism,
        mechanism_samples=args.max_samples or 128,
    )
    calib = collect_tida_outputs(runtime.model, runtime.loaders["train_calib"], runtime.device)
    thresholds = fit_train_calib_thresholds(calib)
    metrics = {
        "raw_fixed": branch_metrics(rows),
        "deploy": {
            "image": branch_metrics(rows, thresholds["image"])["image"],
            "video": branch_metrics(rows, thresholds["video"])["video"],
        },
        "temporal_contribution": temporal_contribution_metrics(rows),
        "geometric_branches_raw_fixed": geometric_branch_metrics(rows),
        "geometric_effectiveness": geometric_temporal_effectiveness_metrics(rows),
        "traffic_action_effectiveness": traffic_action_effectiveness_metrics(rows),
        "trajectory_traffic_effectiveness": trajectory_traffic_effectiveness_metrics(rows),
        "traffic_adaptive_boundary_effectiveness": (
            traffic_adaptive_boundary_effectiveness_metrics(rows, thresholds["video"])
        ),
    }
    atomic_write_json(Path(args.output_dir) / "evaluation.json", metrics)
    print(json.dumps(metrics, default=str), flush=True)


if __name__ == "__main__":
    main()
