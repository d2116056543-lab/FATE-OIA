from __future__ import annotations

"""Read-only MOSAIC branch, transport, factor and gradient diagnostics.

This module deliberately does not train, update thresholds, or write model
state.  It reuses the formal test loader and the exact checkpoint forward,
then reruns only decoder branches with controlled interventions.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex
from fate_oia.datasets.mosaic_grounding_observations import MOSAICGroundingObservationBuilder
from fate_oia.engine.train_acpr_mosaic_ad import (
    _grounding_records,
    build_loaders,
    build_model_components,
    load_config,
)
from fate_oia.metrics import binary_average_precision, multilabel_metrics_from_logits
from fate_oia.utils.mosaic_artifacts import write_json
from fate_oia.utils.mosaic_checkpoint import load_mosaic_model_state_strict


ACTION_ALLOWED_STATES = {
    "forward": {"forward_feasible", "lane_follow_permitted"},
    "stop": {"stop_obligation", "front_risk"},
    "left": {"left_affordance", "left_veto"},
    "right": {"right_affordance", "right_veto"},
}


def binary_roc_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute a finite-safe binary ROC AUC without sklearn dependency."""
    scores = scores.detach().float().flatten()
    targets = (targets.detach().float().flatten() > 0.5)
    positives = int(targets.sum().item())
    negatives = int((~targets).sum().item())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=False, stable=True)
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float32, device=scores.device)
    # ``ranks`` is already in ascending-score order; indexing it by the
    # sorted labels, rather than by original sample indices, gives the true
    # Mann-Whitney rank statistic.
    positive_rank_sum = ranks[targets[order]].sum()
    u = positive_rank_sum - positives * (positives + 1) / 2.0
    return float((u / (positives * negatives)).item())


def _safe_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _metric_view(logits: torch.Tensor, targets: torch.Tensor, prefix: str) -> dict[str, Any]:
    metrics = multilabel_metrics_from_logits(logits, targets, threshold=0.5)
    auc = [binary_roc_auc(logits[:, index], targets[:, index]) for index in range(targets.shape[1])]
    return {
        "mF1": float(metrics[f"{prefix}mF1"]),
        "oF1": float(metrics[f"{prefix}oF1"]),
        "mAP": float(metrics[f"{prefix}mAP"]),
        "per_label_f1": metrics[f"{prefix}per_label_f1"],
        "per_label_ap": metrics[f"{prefix}per_label_ap"],
        "per_label_auc": auc,
    }


def _wrong_flip_counts(
    base_logits: torch.Tensor, deploy_logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, list[int]]:
    base = base_logits >= 0.0
    deploy = deploy_logits >= 0.0
    positive = targets > 0.5
    return {
        # With fixed ground truth, FP->TP and FN->TN are logically zero;
        # retain them in the schema to make the threshold transition explicit.
        "FP_to_TP": [0] * targets.shape[1],
        "TP_to_FN": ((positive & base & ~deploy).sum(0)).tolist(),
        "TN_to_FP": ((~positive & ~base & deploy).sum(0)).tolist(),
        "FN_to_TN": [0] * targets.shape[1],
        "positive_prediction_to_negative": ((positive & base & ~deploy).sum(0)).tolist(),
        "negative_prediction_to_positive": ((~positive & ~base & deploy).sum(0)).tolist(),
    }


def _reason_split_logits(model: nn.Module, output: dict[str, Any]) -> dict[str, torch.Tensor]:
    semantic = output["reason_nodes_semantic"]
    visual = output["reason_nodes_visual"]
    weight = model.reason_decoder.classifier_weight
    bias = model.reason_decoder.classifier_bias
    zero_semantic = torch.zeros_like(semantic)
    zero_visual = torch.zeros_like(visual)
    return {
        "reason_visual_only": torch.einsum(
            "brd,rd->br", torch.cat((zero_semantic, visual), dim=-1), weight
        ) + bias,
        "reason_semantic_only": torch.einsum(
            "brd,rd->br", torch.cat((semantic, zero_visual), dim=-1), weight
        ) + bias,
        "reason_final": output["reason_logits_latent"],
    }


def _branch_forward(
    model: nn.Module,
    output: dict[str, Any],
    *,
    state_mode: str = "full",
    factor_masks: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    intermediates = output["_diagnostic_intermediates"]
    state_prob = output["decision_state_prob"]
    state_uncertainty = output["decision_state_uncertainty"]
    if state_mode == "off":
        state_prob = torch.zeros_like(state_prob)
        state_uncertainty = torch.ones_like(state_uncertainty)
    elif state_mode == "shuffled":
        state_prob = torch.roll(state_prob, shifts=1, dims=0)
        state_uncertainty = torch.roll(state_uncertainty, shifts=1, dims=0)
    elif state_mode != "full":
        raise ValueError(f"unknown state mode {state_mode}")
    action = model.action_decoder(
        intermediates["action_pyramid"],
        state_prob,
        state_uncertainty,
        state_gate_cap=model._action_state_gate_cap_value,
    )
    reason = model.reason_decoder(
        intermediates["reason_pyramid"],
        output["factor_features"],
        output["factor_soft_masks"] if factor_masks is None else factor_masks,
        state_prob,
        state_uncertainty,
        state_contribution_cap=model._reason_state_contribution_cap_value,
    )
    return {**action, **reason}


def _allowed_state_indices(model: nn.Module) -> list[list[int]]:
    names = list(model.schema_bundle["states"])
    return [
        [index for index, name in enumerate(names) if name in ACTION_ALLOWED_STATES[action_name]]
        for action_name in ("forward", "stop", "left", "right")
    ]


def _transport_rows(model: nn.Module, output: dict[str, Any], action_targets: torch.Tensor) -> dict[str, Any]:
    attention = output["action_state_attention"].float()
    gate = output["action_state_gate"].float()
    visual = output["action_logits_visual"].float()
    state = output["action_logits_state"].float()
    delta = gate * state
    entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(-1)
    allowed = _allowed_state_indices(model)
    allowed_mass = torch.stack([attention[:, i].index_select(1, torch.tensor(indices, device=attention.device)).sum(-1) for i, indices in enumerate(allowed)], dim=1)
    rms_delta = delta.square().mean(0).sqrt()
    rms_visual = visual.square().mean(0).sqrt().clamp_min(1e-8)
    ratio = rms_delta / rms_visual
    valid_delta = delta.abs() > 1e-8
    sign_agreement = torch.where(
        valid_delta,
        (delta.sign() == (action_targets.float() * 2.0 - 1.0)).float(),
        torch.full_like(delta, float("nan")),
    )
    return {
        "action_names": ["forward", "stop", "left", "right"],
        "state_names": list(model.schema_bundle["states"]),
        "state_attention_entropy_mean": entropy.mean(0).tolist(),
        "state_attention_max_mean": attention.amax(-1).mean(0).tolist(),
        "allowed_state_attention_mass_mean": allowed_mass.mean(0).tolist(),
        "state_gate_mean": gate.mean(0).tolist(),
        "state_gate_p50": gate.median(0).values.tolist(),
        "state_gate_p95": torch.quantile(gate, 0.95, dim=0).tolist(),
        "rms_gate_state_over_visual": ratio.tolist(),
        "rms_gate_state": rms_delta.tolist(),
        "rms_visual": rms_visual.tolist(),
        "per_action_logit_delta_mean": delta.mean(0).tolist(),
        "per_action_logit_delta_abs_mean": delta.abs().mean(0).tolist(),
        "delta_gt_margin_sign_agreement": [
            float(torch.nanmean(sign_agreement[:, index]).item()) for index in range(4)
        ],
    }


def _reason_contamination(model: nn.Module, output: dict[str, Any]) -> list[dict[str, Any]]:
    factor_masks = output["factor_soft_masks"].float()
    reason_masks = output["reason_factor_masks"].float()
    reason_map = model.reason_decoder.reason_factor_map.to(device=factor_masks.device)
    semantic_attention = output["reason_semantic_attention"].float()
    factor_attention = semantic_attention[:, :, : factor_masks.shape[1]]
    state_attention = semantic_attention[:, :, factor_masks.shape[1] :]
    state_names = list(model.schema_bundle["states"])
    rows = []
    for reason_id, mapping in model.schema_bundle["reason_observation"].items():
        allowed_factors = reason_map[reason_id]
        disallowed_factors = ~allowed_factors
        allowed_factor_mass = factor_masks[:, allowed_factors].mean() if allowed_factors.any() else factor_masks.new_zeros(())
        disallowed_factor_mass = factor_masks[:, disallowed_factors].mean() if disallowed_factors.any() else factor_masks.new_zeros(())
        allowed_states = [state_names.index(name) for name in mapping["support_states"] if name in state_names]
        disallowed_states = [index for index in range(len(state_names)) if index not in allowed_states]
        allowed_state_mass = state_attention[:, reason_id, allowed_states].sum(-1).mean() if allowed_states else state_attention.new_zeros(())
        disallowed_state_mass = state_attention[:, reason_id, disallowed_states].sum(-1).mean() if disallowed_states else state_attention.new_zeros(())
        mask = reason_masks[:, reason_id].flatten(1).clamp_min(1e-8)
        normalized = mask / mask.sum(-1, keepdim=True).clamp_min(1e-8)
        entropy = -(normalized * normalized.log()).sum(-1)
        rows.append({
            "reason_id": int(reason_id),
            "allowed_factor_mass": float(allowed_factor_mass.item()),
            "disallowed_factor_mass": float(disallowed_factor_mass.item()),
            "allowed_state_mass": float(allowed_state_mass.item()),
            "disallowed_state_mass": float(disallowed_state_mass.item()),
            "reason_factor_mask_area": float(mask.mean().item()),
            "reason_factor_mask_entropy": float(entropy.mean().item()),
        })
    return rows


def _masked_ap(scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    selected = mask > 0.5
    if not bool(selected.any()):
        return float("nan")
    return binary_average_precision(scores[selected], targets[selected])


def _factor_quality_rows(
    model: nn.Module,
    outputs: dict[str, dict[str, Any]],
    observations: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    names = [factor["name"] for factor in model.schema_bundle["factors"]]
    presence_target = observations["presence_target"].float()
    presence_mask = observations["presence_mask"].float()
    visibility_target = observations["visibility_target"].float()
    visibility_mask = observations["visibility_mask"].float()
    weak_negative = observations["weak_negative_mask"].float()
    geometry = observations["geometry_mask"].float()
    geometry_valid = observations["geometry_mask_valid"].float()
    full = outputs["full"]
    content = outputs["content_only"]
    prior = outputs["prior_only"]
    shuffled = torch.roll(full["factor_presence_prob"], shifts=1, dims=0)
    rows = []
    for factor_id, factor_name in enumerate(names):
        confirmed = (presence_mask[:, factor_id] > 0.5) & (presence_target[:, factor_id] > 0.5) & ~(weak_negative[:, factor_id] > 0.5)
        weak_neg = (weak_negative[:, factor_id] > 0.5)
        unknown = ~(presence_mask[:, factor_id] > 0.5)
        geom = geometry[:, factor_id]
        pred_mask = full["factor_soft_masks"][:, factor_id]
        intersection = torch.minimum(pred_mask, geom).sum((-2, -1))
        union = torch.maximum(pred_mask, geom).sum((-2, -1)).clamp_min(1e-8)
        soft_iou = intersection[geometry_valid[:, factor_id] > 0.5] / union[geometry_valid[:, factor_id] > 0.5]
        support = presence_mask[:, factor_id]
        rows.append({
            "factor_id": factor_id,
            "factor_name": factor_name,
            "confirmed_positive_count": int(confirmed.sum().item()),
            "weak_negative_count": int(weak_neg.sum().item()),
            "unknown_count": int(unknown.sum().item()),
            "presence_auprc": _masked_ap(full["factor_presence_prob"][:, factor_id], presence_target[:, factor_id], support),
            "visibility_auprc": _masked_ap(full["factor_visibility_prob"][:, factor_id], visibility_target[:, factor_id], visibility_mask[:, factor_id]),
            "soft_iou": float(soft_iou.mean().item()) if soft_iou.numel() else float("nan"),
            "prototype_effective_count": float(full["prototype_effective_count"][:, factor_id].mean().item()),
            "content_only_presence_auprc": _masked_ap(content["factor_presence_prob"][:, factor_id], presence_target[:, factor_id], support),
            "prior_only_presence_auprc": _masked_ap(prior["factor_presence_prob"][:, factor_id], presence_target[:, factor_id], support),
            "content_minus_full_presence_auprc": float(_masked_ap(content["factor_presence_prob"][:, factor_id], presence_target[:, factor_id], support) - _masked_ap(full["factor_presence_prob"][:, factor_id], presence_target[:, factor_id], support)),
            "prior_minus_full_presence_auprc": float(_masked_ap(prior["factor_presence_prob"][:, factor_id], presence_target[:, factor_id], support) - _masked_ap(full["factor_presence_prob"][:, factor_id], presence_target[:, factor_id], support)),
            "query_shuffle_presence_auprc": _masked_ap(shuffled[:, factor_id], presence_target[:, factor_id], support),
        })
    return rows


def _selective_summary(selective: nn.Module, output: dict[str, Any], reasons: torch.Tensor, file_names: list[str], generator: torch.Generator) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = selective(output["reason_logits_latent"], reasons, output["factor_visibility_prob"], output["factor_uncertainty"])
    hidden_observed, hidden_mask = selective.hide_observed_positives(reasons, generator=generator)
    hidden_result = selective(output["reason_logits_latent"], hidden_observed, output["factor_visibility_prob"], output["factor_uncertainty"])
    q = hidden_result["reason_latent_posterior_live"].detach()
    zero_mask = hidden_observed <= 0.5
    top_rows = []
    for batch_index, file_name in enumerate(file_names):
        values = q[batch_index].masked_fill(~zero_mask[batch_index], -1.0)
        count = min(200, int(zero_mask[batch_index].sum().item()))
        if count:
            top = torch.topk(values, count).indices.tolist()
            for reason_id in top:
                top_rows.append({
                    "file_name": file_name,
                    "reason_id": int(reason_id),
                    "posterior_q": float(q[batch_index, reason_id].item()),
                    "synthetic_hidden_positive": bool(hidden_mask[batch_index, reason_id].item()),
                })
    synthetic_targets = hidden_mask.float()
    synthetic_scores = q
    synthetic_auprc = _safe_mean(binary_average_precision(synthetic_scores[:, r], synthetic_targets[:, r]) for r in range(21))
    summary = {
        "observed_positive_rate": float((reasons > 0.5).float().mean().item()),
        "latent_positive_rate": float(torch.sigmoid(output["reason_logits_latent"]).mean().item()),
        "propensity_mean": float(result["reason_propensity"].mean().item()),
        "propensity_min": float(result["reason_propensity"].min().item()),
        "propensity_max": float(result["reason_propensity"].max().item()),
        "propensity_bound_saturation_rate": float(((result["reason_propensity"] <= 0.201) | (result["reason_propensity"] >= 0.949)).float().mean().item()),
        "posterior_q_observed_zero_mean": float(result["reason_latent_posterior_live"].masked_select(reasons <= 0.5).mean().item()),
        "synthetic_hidden_positive_count": int(hidden_mask.sum().item()),
        "synthetic_hidden_positive_auprc": synthetic_auprc,
        "manual_review_required": True,
        "manual_review_note": "Top-q observed-zero rows are exported; precision is synthetic until a human reviews >=200 rows.",
    }
    return summary, top_rows


def _module_parameters(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    return {
        name: [parameter for parameter in getattr(model, name).parameters() if parameter.requires_grad]
        for name in ("visual_pyramid", "action_adapter", "reason_adapter", "observable_predicates", "state_composer", "action_decoder", "reason_decoder")
    }


def _grad_vector(parameters: list[nn.Parameter]) -> torch.Tensor:
    if not parameters:
        return torch.zeros(1)
    values = [
        parameter.grad.detach().flatten()
        if parameter.grad is not None
        else torch.zeros_like(parameter).flatten()
        for parameter in parameters
    ]
    return torch.cat(values)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = left.norm() * right.norm()
    return float((left @ right / denom.clamp_min(1e-12)).item()) if denom.item() > 0 else float("nan")


def _gradient_attribution(model: nn.Module, selective: nn.Module, batch: dict[str, Any], device: torch.device, observations: dict[str, torch.Tensor]) -> dict[str, Any]:
    model.train(False)
    selective.train(False)
    images = batch["image"].to(device)
    actions = batch["action"].to(device)
    reasons = batch["reason"].to(device)
    with torch.enable_grad():
        output = model(images, return_masks=True, return_intermediates=True)
        selective_output = selective(output["reason_logits_latent"], reasons, output["factor_visibility_prob"], output["factor_uncertainty"])
        action_loss = F.binary_cross_entropy_with_logits(output["action_logits_raw"], actions)
        factor_mask = observations["presence_mask"].to(device)
        factor_target = observations["presence_target"].to(device)
        factor_loss = F.binary_cross_entropy_with_logits(output["factor_presence_logits"], factor_target, weight=factor_mask, reduction="sum") / factor_mask.sum().clamp_min(1.0)
        observation_loss = F.binary_cross_entropy(selective_output["reason_observation_prob"], reasons)
        posterior = selective_output["reason_latent_posterior_live"]
        positive = reasons > 0.5
        negative = ~positive
        pair_terms = []
        for reason_id in range(21):
            pos = output["reason_logits_latent"][:, reason_id][positive[:, reason_id]]
            neg = output["reason_logits_latent"][:, reason_id][negative[:, reason_id]]
            if pos.numel() and neg.numel():
                pair_terms.append(F.softplus(-(pos[:, None] - neg[None, :])).mean())
        posterior_ranking_loss = torch.stack(pair_terms).mean() if pair_terms else output["reason_logits_latent"].sum() * 0.0
        losses = {
            "action_loss": action_loss,
            "factor_loss": factor_loss,
            "reason_observation_loss": observation_loss,
            "posterior_ranking_loss": posterior_ranking_loss,
        }
        params = _module_parameters(model)
        captured: dict[str, dict[str, float]] = {}
        vectors: dict[str, dict[str, torch.Tensor]] = {}
        for loss_name, loss in losses.items():
            model.zero_grad(set_to_none=True)
            selective.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            vectors[loss_name] = {name: _grad_vector(values).detach().cpu() for name, values in params.items()}
            captured[loss_name] = {name: float(vector.norm().item()) for name, vector in vectors[loss_name].items()}
        model.zero_grad(set_to_none=True)
        selective.zero_grad(set_to_none=True)
        action_to_reason_adapter = torch.autograd.grad(
            output["action_logits_raw"].sum(), params["reason_adapter"], allow_unused=True, retain_graph=True
        )
        reason_to_action_adapter = torch.autograd.grad(
            output["reason_logits_latent"].sum(), params["action_adapter"], allow_unused=True, retain_graph=True
        )
        firewall = {
            "grad_action_logits_wrt_reason_adapter_norm": float(torch.cat([value.flatten() for value in action_to_reason_adapter if value is not None]).norm().item()) if any(value is not None for value in action_to_reason_adapter) else 0.0,
            "grad_reason_logits_wrt_action_adapter_norm": float(torch.cat([value.flatten() for value in reason_to_action_adapter if value is not None]).norm().item()) if any(value is not None for value in reason_to_action_adapter) else 0.0,
            "reason_observation_loss_action_adapter_norm": captured["reason_observation_loss"]["action_adapter"],
        }
        cosine = {}
        for left_name, right_name in (("action_loss", "reason_observation_loss"), ("action_loss", "factor_loss"), ("action_loss", "posterior_ranking_loss")):
            cosine[f"{left_name}_vs_{right_name}"] = {
                name: _cosine(vectors[left_name][name], vectors[right_name][name]) for name in params
            }
        return {"loss_grad_norms": captured, "gradient_cosines": cosine, "firewall": firewall}


def _load_checkpoint(model: nn.Module, selective: nn.Module, threshold: nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_mosaic_model_state_strict(model, payload["model"])
    selective.load_state_dict(payload["selective_observation"], strict=True)
    threshold.load_state_dict(payload["calibrator"], strict=True)
    return payload


@torch.no_grad()
def _collect_checkpoint(
    model: nn.Module,
    selective: nn.Module,
    threshold: nn.Module,
    loader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    *,
    max_batches: int | None = None,
    grounding_index: BDD100KGroundingIndex | None,
    grounding_builder: MOSAICGroundingObservationBuilder,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    selective.eval()
    threshold.eval()
    names = [factor["name"] for factor in model.schema_bundle["factors"]]
    branches: dict[str, dict[str, list[torch.Tensor]]] = {}
    factor_outputs: dict[str, list[dict[str, torch.Tensor]]] = {name: [] for name in ("full", "content_only", "prior_only")}
    all_actions: list[torch.Tensor] = []
    all_reasons: list[torch.Tensor] = []
    all_files: list[str] = []
    all_observations: list[dict[str, torch.Tensor]] = []
    transport_parts: list[dict[str, torch.Tensor]] = []
    contamination_parts: list[list[dict[str, Any]]] = []
    selective_parts: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    first_batch_for_grad: dict[str, Any] | None = None
    first_observation: dict[str, torch.Tensor] | None = None
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        reasons = batch["reason"].to(device, non_blocking=True)
        files = [str(value) for value in batch["file_name"]]
        output = model(images, return_masks=True, return_intermediates=True)
        full_branch = _branch_forward(model, output)
        state_off = _branch_forward(model, output, state_mode="off")
        state_shuffled = _branch_forward(model, output, state_mode="shuffled")
        factor_shuffled = _branch_forward(model, output, factor_masks=torch.roll(output["factor_soft_masks"], shifts=1, dims=0))
        semantic_visual = _reason_split_logits(model, output)
        variant_values = {
            "visual_only": {"action": output["action_logits_visual"], "reason": semantic_visual["reason_visual_only"]},
            "state_only": {"action": output["action_logits_state"], "reason": semantic_visual["reason_semantic_only"]},
            "final": {"action": output["action_logits_raw"], "reason": semantic_visual["reason_final"]},
            "state_off": {"action": state_off["action_logits_raw"], "reason": state_off["reason_logits_latent"]},
            "state_shuffled": {"action": state_shuffled["action_logits_raw"], "reason": state_shuffled["reason_logits_latent"]},
            "factor_masks_shuffled": {"action": factor_shuffled["action_logits_raw"], "reason": factor_shuffled["reason_logits_latent"]},
        }
        threshold_output = threshold(output["action_logits_raw"], output["reason_logits_latent"])
        variant_values["final_deploy"] = {
            "action": threshold_output["action_logits_deploy"],
            "reason": threshold_output["reason_logits_deploy"],
        }
        for prior_mode in ("content_only", "prior_only"):
            prior_output = model(images, prior_mode=prior_mode, return_masks=True)
            variant_values[prior_mode] = {"action": prior_output["action_logits_raw"], "reason": prior_output["reason_logits_latent"]}
            factor_outputs[prior_mode].append({
                "factor_presence_prob": prior_output["factor_presence_prob"].detach().cpu(),
                "factor_visibility_prob": prior_output["factor_visibility_prob"].detach().cpu(),
            })
        factor_outputs["full"].append({
            "factor_presence_prob": output["factor_presence_prob"].detach().cpu(),
            "factor_visibility_prob": output["factor_visibility_prob"].detach().cpu(),
            "factor_soft_masks": output["factor_soft_masks"].detach().cpu(),
            "prototype_effective_count": output["measurement_stats"]["prototype_effective_count"].detach().cpu().expand(images.shape[0], -1),
        })
        for name, values in variant_values.items():
            branches.setdefault(name, {"action": [], "reason": []})
            branches[name]["action"].append(values["action"].detach().float().cpu())
            branches[name]["reason"].append(values["reason"].detach().float().cpu())
        all_actions.append(actions.detach().float().cpu())
        all_reasons.append(reasons.detach().float().cpu())
        all_files.extend(files)
        transport_parts.append({name: value.detach().float().cpu() for name, value in {
            "attention": output["action_state_attention"], "gate": output["action_state_gate"],
            "visual": output["action_logits_visual"], "state": output["action_logits_state"],
        }.items()})
        contamination_parts.append(_reason_contamination(model, output))
        if first_batch_for_grad is None:
            first_batch_for_grad = {name: value for name, value in batch.items() if name in {"image", "action", "reason", "file_name"}}
            if grounding_index is not None:
                records = _grounding_records(grounding_index, files)
                first_observation = grounding_builder(reasons, records, split="train")
            else:
                first_observation = {
                    "presence_target": torch.zeros(images.shape[0], len(names), device=device),
                    "presence_mask": torch.zeros(images.shape[0], len(names), device=device),
                    "visibility_target": torch.zeros(images.shape[0], len(names), device=device),
                    "visibility_mask": torch.zeros(images.shape[0], len(names), device=device),
                    "weak_negative_mask": torch.zeros(images.shape[0], len(names), device=device),
                    "geometry_mask": torch.zeros(images.shape[0], len(names), 45, 80, device=device),
                    "geometry_mask_valid": torch.zeros(images.shape[0], len(names), device=device),
                }
        if grounding_index is not None:
            records = _grounding_records(grounding_index, files)
            all_observations.append(grounding_builder(reasons, records, split="train"))
        selective_summary, rows = _selective_summary(selective, output, reasons, files, torch.Generator(device=device).manual_seed(seed + batch_index))
        selective_parts.append(selective_summary)
        review_rows.extend(rows)
    if not all_actions:
        raise ValueError("diagnostic loader produced no batches")
    actions = torch.cat(all_actions)
    reasons = torch.cat(all_reasons)
    branch_logits = {name: {kind: torch.cat(values) for kind, values in branches[name].items()} for name in branches}
    metrics = {
        name: {"action": _metric_view(values["action"], actions, ""), "reason": _metric_view(values["reason"], reasons, "")}
        for name, values in branch_logits.items()
    }
    deploy_delta = {
        "action_mean_logit_delta_by_label": (
            branch_logits["final_deploy"]["action"] - branch_logits["final"]["action"]
        ).mean(0).tolist(),
        "reason_mean_logit_delta_by_label": (
            branch_logits["final_deploy"]["reason"] - branch_logits["final"]["reason"]
        ).mean(0).tolist(),
        "action_wrong_flip_counts": _wrong_flip_counts(
            branch_logits["final"]["action"], branch_logits["final_deploy"]["action"], actions
        ),
        "reason_wrong_flip_counts": _wrong_flip_counts(
            branch_logits["final"]["reason"], branch_logits["final_deploy"]["reason"], reasons
        ),
    }
    transport = {
        "batches": len(transport_parts),
        "attention": torch.cat([part["attention"] for part in transport_parts]),
        "gate": torch.cat([part["gate"] for part in transport_parts]),
        "visual": torch.cat([part["visual"] for part in transport_parts]),
        "state": torch.cat([part["state"] for part in transport_parts]),
    }
    transport_output = _transport_rows(model, {
        "action_state_attention": transport["attention"], "action_state_gate": transport["gate"],
        "action_logits_visual": transport["visual"], "action_logits_state": transport["state"],
    }, actions)
    contamination = []
    for reason_id in range(21):
        values = [row[reason_id] for row in contamination_parts]
        contamination.append({key: _safe_mean(item[key] for item in values) for key in values[0]})
    full_factor = {key: torch.cat([item[key] for item in factor_outputs["full"]]) for key in factor_outputs["full"][0]}
    mode_factor = {
        mode: {key: torch.cat([item[key] for item in factor_outputs[mode]]) for key in factor_outputs[mode][0]}
        for mode in ("content_only", "prior_only")
    }
    if grounding_index is not None:
        observations = {key: torch.cat([item[key].detach().cpu() for item in all_observations if key in item]) for key in all_observations[0]}
    else:
        observations = {"presence_target": torch.zeros(actions.shape[0], len(names)), "presence_mask": torch.zeros(actions.shape[0], len(names)), "visibility_target": torch.zeros(actions.shape[0], len(names)), "visibility_mask": torch.zeros(actions.shape[0], len(names)), "weak_negative_mask": torch.zeros(actions.shape[0], len(names)), "geometry_mask": torch.zeros(actions.shape[0], len(names), 45, 80), "geometry_mask_valid": torch.zeros(actions.shape[0], len(names))}
    factor_rows = _factor_quality_rows(model, {
        "full": {**full_factor, "factor_presence_prob": full_factor["factor_presence_prob"]},
        "content_only": mode_factor["content_only"], "prior_only": mode_factor["prior_only"],
    }, observations)
    gradient = _gradient_attribution(
        model,
        selective,
        first_batch_for_grad,
        device,
        {key: value.to(device) for key, value in (first_observation or {}).items()},
    ) if first_batch_for_grad is not None and first_observation is not None else {"available": False}
    artifact_dir = output_dir / f"epoch_{epoch}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"branches": branch_logits, "labels_action": actions, "labels_reason": reasons, "file_names": all_files}, artifact_dir / "branch_logits_and_labels.pt")
    write_json(artifact_dir / "metrics_by_branch.json", metrics)
    write_json(artifact_dir / "action_transport.json", transport_output)
    write_json(artifact_dir / "reason_contamination.json", contamination)
    write_json(artifact_dir / "factor_quality.json", factor_rows)
    write_json(artifact_dir / "selective_observation.json", selective_parts)
    write_json(artifact_dir / "gradient_attribution.json", gradient)
    write_json(artifact_dir / "deploy_vs_base_deltas.json", deploy_delta)
    with (artifact_dir / "manual_review_top_q_observed_zero.jsonl").open("w", encoding="utf-8") as stream:
        for row in review_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "epoch": epoch,
        "sample_count": int(actions.shape[0]),
        "metrics_by_branch": metrics,
        "action_transport": transport_output,
        "reason_contamination": contamination,
        "factor_quality": factor_rows,
        "selective_observation": selective_parts,
        "gradient_attribution": gradient,
        "deploy_vs_base": deploy_delta,
        "artifacts": str(artifact_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="epoch=path; repeat for each checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--max_batches", type=int)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--strong_baseline_checkpoint")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    device = torch.device(args.device)
    _, _, test_loader, split_stats = build_loaders(
        config, output_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        max_test_samples=args.max_test_samples, seed=args.seed,
    )
    model, selective, threshold, _, _ = build_model_components(config, args.config, device)
    index = BDD100KGroundingIndex(config["data"]["bdd100k_root"])
    builder = MOSAICGroundingObservationBuilder(model.schema_bundle["factors"])
    available = {}
    for item in args.checkpoint:
        raw_epoch, raw_path = item.split("=", 1)
        epoch = int(raw_epoch)
        checkpoint_path = Path(raw_path)
        if not checkpoint_path.exists():
            available[str(epoch)] = {"available": False, "reason": "checkpoint_missing", "path": str(checkpoint_path)}
            continue
        payload = _load_checkpoint(model, selective, threshold, checkpoint_path, device)
        actual_epoch = int(payload.get("epoch", epoch))
        available[str(epoch)] = _collect_checkpoint(
            model, selective, threshold, test_loader, device, output_dir, actual_epoch,
            max_batches=args.max_batches, grounding_index=index, grounding_builder=builder, seed=args.seed,
        )
    baseline = {
        "available": False,
        "reason": "RunC checkpoint uses an incompatible model contract and was not loaded into MOSAIC",
        "path": args.strong_baseline_checkpoint,
    }
    summary = {
        "diagnostic": "D1 same-checkpoint branch isolation + D3 gradient attribution",
        "training_started": False,
        "split": "test",
        "test_threshold_writeback": False,
        "grounding_note": "BDD100K observations are test-image diagnostics only; they do not alter logits or thresholds.",
        "split_stats": split_stats,
        "checkpoints": available,
        "strong_baseline": baseline,
        "required_followup": {
            "D2": "short resume variants are not run by this read-only diagnostic entry; run only after D1 identifies a causal branch",
            "manual_review": "review at least 200 exported observed-zero/high-q rows before claiming posterior recovery",
        },
    }
    write_json(output_dir / "diagnostic_summary.json", summary)


if __name__ == "__main__":
    main()
