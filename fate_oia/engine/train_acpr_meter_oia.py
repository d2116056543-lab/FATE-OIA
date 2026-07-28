from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.meter_dataset import METERDataset, fixed_meter_split_indices, meter_split_manifest
from fate_oia.datasets.meter_grounding_index import METERGroundingIndex
from fate_oia.engine.eval_acpr_meter_oia import collect_outputs, metrics_summary, branch_metrics, mechanism_stats_from_collected
from fate_oia.losses.meter_action_losses import meter_action_loss
from fate_oia.losses.meter_counterfactual_losses import meter_counterfactual_loss
from fate_oia.losses.meter_grounding_losses import meter_grounding_loss
from fate_oia.losses.meter_pu_losses import meter_hidden_positive_audit, meter_private_pu_loss, meter_pu_score
from fate_oia.losses.meter_reason_losses import meter_reason_loss
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.optim.meter_meta_utility import METERMetaUtility
from fate_oia.transforms_meter import meter_image_transform
from fate_oia.utils.meter_artifacts import (
    append_jsonl,
    combined_file_hash,
    file_hash,
    python_source_tree_hash,
    save_checkpoint,
    save_epoch_artifacts,
    state_hash,
    write_json,
    load_checkpoint,
)
from fate_oia.utils.meter_config import load_meter_config
from fate_oia.utils.meter_posthoc_calibration import METERCalibrationResult, apply_meter_deploy, fit_train_calib_deploy_theta, guard_train_calib_deploy_theta


FULL_READY_REQUIRED_CHECKS = {
    "exact_pilot_protocol",
    "finite",
    "all_owner_steps_positive",
    "peak_reserved_under_45gb",
    "action_semantic_compatible",
    "action_final_compatible",
    "semantic_visual_ratio",
    "selector_nonconstant",
    "selector_regret_decreased",
    "reason_mix_synergy",
    "reason_final_not_below_calalign",
    "annotation_residual_range",
    "factor_shuffle_hurts_12_labels",
    "meta_high_omega_4",
    "meta_low_omega_4",
    "meta_share_rate",
    "meta_positive_utility",
    "evidence_not_max_entropy",
    "support_null_nontrivial",
    "counter_null_nontrivial",
    "counterfactual_all_actions",
    "counterfactual_12_factors",
    "selected_beats_control",
    "counter_direction",
    "github_head_matches",
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _loader(
    dataset: METERDataset,
    indices: list[int],
    cfg: dict[str, Any],
    *,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    data_cfg = cfg["data"]
    workers = int(data_cfg.get("num_workers", 4))
    kwargs: dict[str, Any] = {
        "batch_size": int(cfg["training"].get("batch_size", 6)),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "persistent_workers": bool(data_cfg.get("persistent_workers", True)) and workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    if seed is not None:
        kwargs["generator"] = torch.Generator().manual_seed(int(seed))
    return DataLoader(Subset(dataset, indices), **kwargs)


def _move_grounding(batch: dict[str, Any], device: torch.device, reason_dim: int = 21) -> dict[str, Tensor]:
    raw = batch.get("meter_grounding")
    if raw is None:
        raise RuntimeError("METER training batch is missing signed grounding targets")
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in raw.items()}


def _slice_batch_tree(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index : index + 1]
    if isinstance(value, dict):
        return {key: _slice_batch_tree(item, index, batch_size) for key, item in value.items()}
    return value


def _select_mass_mask(
    score_maps: Tensor,
    factor_ids: Tensor,
    *,
    selected_mass: float,
    minimum_patches: int,
    max_patches: int,
) -> tuple[Tensor, Tensor]:
    if score_maps.ndim != 3:
        raise ValueError("score_maps must be [B,F,N]")
    batch, _, patches = score_maps.shape
    mask = torch.zeros(batch, patches, dtype=torch.bool, device=score_maps.device)
    counts = torch.zeros(batch, dtype=torch.long, device=score_maps.device)
    mass = float(min(max(selected_mass, 0.0), 1.0))
    for row in range(batch):
        scores = score_maps[row, int(factor_ids[row])].detach().float().clamp_min(0.0)
        order = torch.argsort(scores, descending=True)
        total = scores.sum()
        if float(total) <= 0.0:
            count = int(minimum_patches)
        else:
            cumulative = scores[order].cumsum(dim=0) / total
            count = int(torch.searchsorted(cumulative, cumulative.new_tensor(mass), right=False).item()) + 1
        count = min(max(count, int(minimum_patches)), int(max_patches), patches - 1)
        counts[row] = count
        mask[row, order[:count]] = True
    return mask, counts


def _matched_control_mask(
    selected_mask: Tensor,
    response_maps: Tensor,
    factor_ids: Tensor,
    token_norm: Tensor,
    *,
    grid_hw: tuple[int, int] = (45, 80),
) -> tuple[Tensor, int]:
    batch, patches = selected_mask.shape
    height, width = grid_hw
    if patches != height * width:
        raise ValueError("selected mask does not match the patch grid")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=selected_mask.device),
        torch.arange(width, device=selected_mask.device),
        indexing="ij",
    )
    y_flat = yy.flatten()
    x_flat = xx.flatten()
    controls = torch.zeros_like(selected_mask)
    relaxed = 0
    sector_width = max(width // 3, 1)
    vertical_tolerance = max(height // 6, 2)
    for row in range(batch):
        selected = torch.where(selected_mask[row])[0]
        count = int(selected.numel())
        if count == 0:
            continue
        center_y = y_flat[selected].float().mean()
        center_x = x_flat[selected].float().mean()
        sector = torch.div(center_x.long(), sector_width, rounding_mode="floor").clamp_max(2)
        available = ~selected_mask[row]
        same_sector = torch.div(x_flat, sector_width, rounding_mode="floor").clamp_max(2) == sector
        similar_vertical = (y_flat.float() - center_y).abs() <= vertical_tolerance
        candidate = available & same_sector & similar_vertical
        if int(candidate.sum()) < count:
            candidate = available & same_sector
            relaxed += 1
        if int(candidate.sum()) < count:
            candidate = available
            relaxed += 1
        selected_norm = token_norm[row, selected].mean()
        norm_delta = (token_norm[row] - selected_norm).abs()
        response = response_maps[row, int(factor_ids[row])].detach().float()
        spatial = 0.02 * (y_flat.float() - center_y).abs() + 0.005 * (x_flat.float() - center_x).abs()
        score = norm_delta + 2.0 * response + spatial
        score = score.masked_fill(~candidate, float("inf"))
        chosen = torch.topk(score, k=count, largest=False).indices
        controls[row, chosen] = True
    return controls, relaxed


def _replace_selected_with_neighbor_mean(
    tokens: Tensor,
    selected_mask: Tensor,
    *,
    grid_hw: tuple[int, int] = (45, 80),
) -> Tensor:
    batch, patches, dim = tokens.shape
    height, width = grid_hw
    if patches != height * width or selected_mask.shape != (batch, patches):
        raise ValueError("token/mask shape does not match the patch grid")
    image = tokens.transpose(1, 2).reshape(batch, dim, height, width)
    valid = (~selected_mask).reshape(batch, 1, height, width).to(dtype=tokens.dtype)

    def local_mean(kernel: int) -> tuple[Tensor, Tensor]:
        area = float(kernel * kernel)
        summed = F.avg_pool2d(image * valid, kernel_size=kernel, stride=1, padding=kernel // 2) * area
        count = F.avg_pool2d(valid, kernel_size=kernel, stride=1, padding=kernel // 2) * area
        return summed / count.clamp_min(1.0), count

    mean3, count3 = local_mean(3)
    mean5, count5 = local_mean(5)
    global_mean = (image * valid).sum(dim=(2, 3), keepdim=True) / valid.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    replacement = torch.where(count3 > 0, mean3, torch.where(count5 > 0, mean5, global_mean))
    replacement = replacement.reshape(batch, dim, patches).transpose(1, 2)
    return torch.where(selected_mask.unsqueeze(-1), replacement, tokens)


def _counterfactual_event(
    model: METEROIAModel,
    field: dict[str, Any],
    output: dict[str, Any],
    progress: float,
    *,
    action_target: Tensor | None = None,
    selected_mass: float = 0.60,
    max_patches: int = 28,
    minimum_patches: int = 4,
    sample_index: int = 0,
) -> dict[str, Tensor | int | float]:
    """One-sample, same-field selected/control deletion with no DINO rerun."""
    full_batch = output["factor_support_map"].shape[0]
    if full_batch == 0:
        zero = output["factor_support_score"].new_zeros(())
        return {"valid_count": 0, "selected_effect": zero, "control_effect": zero, "total": zero}
    event_row = int(sample_index) % full_batch
    field = _slice_batch_tree(field, event_row, full_batch)
    output = _slice_batch_tree(output, event_row, full_batch)
    if action_target is not None:
        action_target = action_target[event_row : event_row + 1]

    maps = output["factor_support_map"].detach()
    batch, factors, patches = maps.shape
    contributions = output["action_factor_contributions"].detach()
    if action_target is None:
        target_action_id = output["action_logits_final"].detach().argmax(dim=1)
    else:
        positive = action_target.detach().float() > 0.5
        action_strength = contributions.abs().sum(dim=-1).masked_fill(~positive, float("-inf"))
        empty = ~positive.any(dim=1)
        fallback = output["action_logits_final"].detach().argmax(dim=1)
        target_action_id = torch.where(empty, fallback, action_strength.argmax(dim=1))
    row = torch.arange(batch, device=maps.device)
    target_contributions = contributions[row, target_action_id]
    factor_priority = target_contributions.abs() * (0.25 + output["factor_reliability"].detach())
    chosen_factor = factor_priority.argmax(dim=1)
    wrong_factor = factor_priority.argmin(dim=1)

    support_mask, support_counts = _select_mass_mask(
        maps,
        chosen_factor,
        selected_mass=selected_mass,
        minimum_patches=minimum_patches,
        max_patches=max_patches,
    )
    counter_mask, counter_counts = _select_mass_mask(
        output["factor_counter_map"].detach(),
        chosen_factor,
        selected_mass=selected_mass,
        minimum_patches=minimum_patches,
        max_patches=max_patches,
    )
    wrong_factor_mask, _ = _select_mass_mask(
        maps,
        wrong_factor,
        selected_mass=selected_mass,
        minimum_patches=minimum_patches,
        max_patches=max_patches,
    )
    valid = (
        (output["factor_support_null"].detach()[row, chosen_factor] < 0.90)
        & (output["factor_counter_null"].detach()[row, chosen_factor] < 0.90)
    )
    if not bool(valid.any()):
        zero = output["factor_support_score"].new_zeros(())
        return {
            "valid_count": 0,
            "selected_effect": zero,
            "control_effect": zero,
            "total": zero,
            "skip_reason": "high_null_mass",
        }

    token_mean = field["patch_tokens_by_layer"].mean(dim=1)
    token_norm = token_mean.float().square().mean(dim=-1).sqrt()
    response = maps + output["factor_counter_map"].detach()
    support_control_mask, support_control_relaxed = _matched_control_mask(
        support_mask,
        response,
        chosen_factor,
        token_norm,
    )
    counter_control_mask, counter_control_relaxed = _matched_control_mask(
        counter_mask,
        response,
        chosen_factor,
        token_norm,
    )

    def delete_field(mask: Tensor) -> dict[str, Any]:
        patch = field["patch_tokens_by_layer"]
        deleted_layers: list[Tensor] = []
        for layer in range(patch.shape[1]):
            deleted_layers.append(
                _replace_selected_with_neighbor_mean(patch[:, layer], mask)
            )
        result = dict(field)
        result["patch_tokens_by_layer"] = torch.stack(deleted_layers, dim=1)
        return result

    selected_output = model.decode_from_field(delete_field(support_mask), progress=progress)
    control_output = model.decode_from_field(delete_field(support_control_mask), progress=progress)
    counter_output = model.decode_from_field(delete_field(counter_mask), progress=progress)
    counter_control_output = model.decode_from_field(delete_field(counter_control_mask), progress=progress)
    wrong_output = model.decode_from_field(delete_field(wrong_factor_mask), progress=progress)
    factor = chosen_factor
    selected_effect = output["factor_support_score"][row, factor] - selected_output["factor_support_score"][row, factor]
    control_effect = output["factor_support_score"][row, factor] - control_output["factor_support_score"][row, factor]
    wrong_effect = output["factor_support_score"][row, factor] - wrong_output["factor_support_score"][row, factor]
    counter_effect = output["factor_counter_score"][row, factor] - counter_output["factor_counter_score"][row, factor]
    counter_control_effect = output["factor_counter_score"][row, factor] - counter_control_output["factor_counter_score"][row, factor]
    action_count = output["action_logits_final"].shape[1]
    target_mask = F.one_hot(target_action_id, num_classes=action_count).to(dtype=torch.bool)
    wrong_action_mask = ~target_mask

    def masked_action_effect(original: Tensor, deleted: Tensor, mask: Tensor) -> Tensor:
        difference = original - deleted
        return (difference * mask.to(dtype=difference.dtype)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(dtype=difference.dtype)

    target_action_effect = masked_action_effect(output["action_logits_final"], selected_output["action_logits_final"], target_mask)
    control_action_effect = masked_action_effect(output["action_logits_final"], control_output["action_logits_final"], target_mask)
    wrong_action_effect = masked_action_effect(output["action_logits_final"], selected_output["action_logits_final"], wrong_action_mask)
    event = meter_counterfactual_loss(
        selected_effect[valid],
        control_effect[valid],
        wrong_effect[valid],
        selected_effect[valid],
        counter_effect[valid],
        target_action_effect=target_action_effect,
        wrong_action_effect=wrong_action_effect,
    )
    return {
        **event,
        "valid_count": int(valid.sum().item()),
        "sample_index": event_row,
        "target_action_id": int(target_action_id[0].item()),
        "chosen_factor_id": int(chosen_factor[0].item()),
        "wrong_factor_id": int(wrong_factor[0].item()),
        "selected_patch_count": int(support_counts.sum().item()),
        "counter_patch_count": int(counter_counts.sum().item()),
        "support_control_effect": control_effect.detach(),
        "support_selected_effect": selected_effect.detach(),
        "counter_selected_effect": counter_effect.detach(),
        "counter_control_effect": counter_control_effect.detach(),
        "target_action_effect": target_action_effect.detach(),
        "control_action_effect": control_action_effect.detach(),
        "wrong_action_effect": wrong_action_effect.detach(),
        "selected_control_overlap": int((support_mask & support_control_mask).sum().item()),
        "counter_control_overlap": int((counter_mask & counter_control_mask).sum().item()),
        "support_control_relaxed": int(support_control_relaxed),
        "counter_control_relaxed": int(counter_control_relaxed),
    }


def _make_optimizer(model: METEROIAModel, cfg: dict[str, Any]) -> AdamW:
    groups: list[dict[str, Any]] = []
    training = cfg["training"]
    effective_batch = int(training.get("batch_size", 6)) * int(
        training.get("gradient_accumulation_steps", 1)
    )
    reference_batch = int(training.get("reference_effective_batch", 32))
    learning_rate_scale = effective_batch / max(reference_batch, 1)
    weight_decay = float(training.get("weight_decay", 0.05))

    def uses_decay(parameter_name: str, parameter: Tensor) -> bool:
        leaf = parameter_name.rsplit(".", 1)[-1]
        learned_embedding = leaf.endswith("_queries") or "embedding" in parameter_name or leaf in {
            "label_queries",
            "private_queries",
            "factor_value",
            "null_key",
            "support_query",
            "counter_query",
        }
        return parameter.ndim > 1 and not parameter_name.endswith(".bias") and not learned_embedding

    def append_owner(owner: str, lr: float, named_parameters: list[tuple[str, Tensor]]) -> None:
        for decay in (True, False):
            parameters = [
                parameter
                for parameter_name, parameter in named_parameters
                if uses_decay(parameter_name, parameter) is decay
            ]
            if parameters:
                groups.append({
                    "params": parameters,
                    "lr": lr * learning_rate_scale,
                    "weight_decay": weight_decay if decay else 0.0,
                    "name": owner,
                    "decay": decay,
                })

    def rule(name: str, lr_key: str, predicate: Any) -> tuple[str, str, Any]:
        return name, lr_key, predicate
    rules = [
        rule("foundation", "lr_foundation_core", lambda n: n.startswith("foundation.")),
        rule("factor_evidence", "lr_factor_evidence", lambda n: n.startswith("signed_factors.") and not n.startswith("signed_factors.meta_adapters.")),
        rule("semantic_action", "lr_semantic_action", lambda n: n.startswith("action_peer.") and not n.startswith("action_peer.selector.")),
        rule("action_selector", "lr_action_selector", lambda n: n.startswith("action_peer.selector.")),
        rule("reason_global_private", "lr_reason_global_private", lambda n: n.startswith("reason_decoder.") and any(f"reason_decoder.{x}" in n for x in ("private_queries", "layer_router", "global_query", "global_key", "global_value", "global_norm", "reason_self_attention", "reason_self_norm", "global_head"))),
        rule("reason_local_private", "lr_reason_local_private", lambda n: n.startswith("reason_decoder.") and any(f"reason_decoder.{x}" in n for x in ("local_proj", "local_norm", "factor_proj", "action_proj", "local_head", "mix_gate"))),
        rule("reason_annotation", "lr_reason_annotation", lambda n: n.startswith("reason_decoder.annotation_head.")),
        rule("meta_adapter", "lr_meta_adapters", lambda n: n.startswith("signed_factors.meta_adapters.")),
        rule("pu_private", "lr_pu_private", lambda n: n == "reason_decoder.tail_gain"),
    ]
    assigned: set[int] = set()
    for owner, lr_key, predicate in rules:
        parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if predicate(name) and parameter.requires_grad and id(parameter) not in assigned
        ]
        for _, p in parameters:
            assigned.add(id(p))
        if parameters:
            append_owner(owner, float(training.get(lr_key, 2e-4)), parameters)
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in assigned:
            assigned.add(id(p))
            append_owner("unclassified", float(training.get("lr_factor_evidence", 2e-4)), [(name, p)])
    return AdamW(groups)


def _losses(
    model: METEROIAModel,
    output: dict[str, Any],
    batch: dict[str, Any],
    cfg: dict[str, Any],
    progress: float,
    device: torch.device,
    *,
    pu_lambda: Tensor,
    counterfactual: dict[str, Any] | None = None,
    counterfactual_event_scale: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Any]]:
    action_target = batch["action"].to(device, non_blocking=True)
    reason_target = batch["reason"].to(device, non_blocking=True)
    weights = cfg["loss_weights"]
    action = meter_action_loss(output, action_target, weights)
    grounding_target = _move_grounding(batch, device)
    confidence = output["factor_reliability"].detach()
    observability = grounding_target["factor_observability"].detach()
    reason = meter_reason_loss(
        output,
        reason_target,
        confidence,
        weights,
        observability=observability,
    )
    grounding = meter_grounding_loss(output, grounding_target)
    private_probability = torch.sigmoid(output["reason_logits_global"].detach())
    factor_probability = torch.sigmoid(
        output["factor_support_score"].detach() - output["factor_counter_score"].detach()
    )
    pu_score = meter_pu_score(private_probability, factor_probability, 1.0 - output["factor_uncertainty"].detach(), observability)
    pu_lambda = pu_lambda.to(device=device, dtype=reason_target.dtype)
    pu = meter_private_pu_loss(output["reason_logits_pu_private"], reason_target, pu_score, pu_lambda)
    grounding_ramp_fraction = float(weights.get("grounding_ramp_fraction", 0.05))
    counterfactual_ramp_fraction = float(weights.get("counterfactual_ramp_fraction", 0.10))
    grounding_ramp = min(progress / max(grounding_ramp_fraction, 1e-6), 1.0)
    counterfactual_ramp = min(progress / max(counterfactual_ramp_fraction, 1e-6), 1.0)
    cf = counterfactual or {
        "selected_control": output["action_logits_final"].new_zeros(()),
        "specificity": output["action_logits_final"].new_zeros(()),
        "direction": output["action_logits_final"].new_zeros(()),
        "total": output["action_logits_final"].new_zeros(()),
        "valid_count": 0,
    }
    total = (
        action["total"]
        + reason["total"]
        + (
            float(weights.get("grounding_start", 0.02))
            + grounding_ramp
            * (
                float(weights.get("grounding_end", 0.10))
                - float(weights.get("grounding_start", 0.02))
            )
        )
        * grounding["total"]
        + pu
        + float(weights.get("counterfactual_end", 0.03))
        * counterfactual_ramp
        * float(counterfactual_event_scale)
        * cf["total"]
    )
    parts = {
        "action": action["total"],
        "reason": reason["total"],
        "grounding": grounding["total"],
        "pu": pu,
        "counterfactual": cf["total"],
        "total": total,
    }
    diagnostics = {"action": action, "reason": reason, "grounding": grounding, "pu": pu, "counterfactual": cf, "pu_score": pu_score}
    return total, parts, diagnostics


def _fit_calibration(
    model: METEROIAModel,
    loader: DataLoader,
    device: torch.device,
    progress: float,
    *,
    fallback_on_deploy_degradation: bool = True,
) -> METERCalibrationResult:
    collected = collect_outputs(model, loader, device, progress=progress)
    action = fit_train_calib_deploy_theta(
        collected["action"]["final"],
        collected["labels_action"],
        model_state_hash=state_hash(model),
        label_groups=(0, 0, 1, 1),
    )
    factor_schema = yaml.safe_load(Path("configs/meter_factor_schema.yaml").read_text(encoding="utf-8"))
    group_names = [str(item["group"]) for item in factor_schema["factors"]]
    group_lookup = {name: index for index, name in enumerate(dict.fromkeys(group_names))}
    reason = fit_train_calib_deploy_theta(
        collected["reason"]["final"],
        collected["labels_reason"],
        model_state_hash=state_hash(model),
        label_groups=tuple(group_lookup[name] for name in group_names),
    )
    candidate = METERCalibrationResult(
        theta=torch.cat([action.theta, reason.theta]),
        temperature=torch.cat([action.temperature, reason.temperature]),
        strategy=f"action:{action.strategy};reason:{reason.strategy}",
        fallback_theta=torch.cat([action.fallback_theta, reason.fallback_theta]),
        fallback_temperature=torch.cat([action.fallback_temperature, reason.fallback_temperature]),
        model_state_hash_before=action.model_state_hash_before,
        model_state_hash_after=reason.model_state_hash_after,
        fit_split="train_calib",
        representation_updated=False,
    )
    return guard_train_calib_deploy_theta(
        collected["action"]["final"],
        collected["labels_action"],
        collected["reason"]["final"],
        collected["labels_reason"],
        candidate,
        fallback_on_deploy_degradation=fallback_on_deploy_degradation,
    )


def _write_full_train_ready_if_eligible(
    *,
    root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    git_head: str,
    git_branch: str,
    config_hash: str,
    schema_hash: str,
    source_tree_hash: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_pilot = (
        int(args.epochs) == 3
        and int(args.max_train_samples) == 4096
        and int(args.max_audit_samples) == 1024
        and int(args.max_calib_samples) == 512
        and int(args.max_test_samples) == 512
    )
    checks: dict[str, bool] = {"exact_pilot_protocol": exact_pilot}
    if not history:
        report = {"pass": False, "checks": checks, "reason": "missing_pilot_history"}
        write_json(output_dir / "pilot_gate_report.json", report)
        return report
    latest = history[-1]
    branches = latest["branches"]
    factor_stats = latest["factor_stats"]
    reason_stats = latest["reason_view_stats"]
    selector_stats = latest["selector_stats"]
    cf = latest["counterfactual"]

    action_visual = branches["action_visual"]
    action_semantic = branches["action_semantic"]
    action_final = branches["action_final"]
    reason_global = branches["reason_global_private"]
    reason_local = branches["reason_local_private"]
    reason_mix = branches["reason_mix_private"]
    reason_final = branches["reason_final"]
    reason_calalign = branches["reason_calalign_reason"]
    reason_shuffle = branches["reason_map_shuffle"]
    final_ap = reason_final["Exp_per_label_ap"]
    shuffle_ap = reason_shuffle["Exp_per_label_ap"]

    checks.update({
        "finite": bool(latest["finite"]),
        "all_owner_steps_positive": all(int(value) > 0 for value in latest["owner_step_counts"].values()),
        "peak_reserved_under_45gb": float(latest["peak_reserved_gb"]) < 45.0,
        "action_semantic_compatible": float(action_semantic["Act_mAP"]) >= float(action_visual["Act_mAP"]) - 0.010,
        "action_final_compatible": float(action_final["Act_mAP"]) >= max(float(action_visual["Act_mAP"]), float(action_semantic["Act_mAP"])) - 0.002,
        "semantic_visual_ratio": 0.03 <= float(selector_stats["semantic_visual_rms_ratio"]) <= 0.30,
        "selector_nonconstant": float(selector_stats["selector_std"]) > 0.02 and float(selector_stats["selector_min"]) < 0.99 and float(selector_stats["selector_max"]) > 0.01,
        "selector_regret_decreased": len(history) >= 2 and float(history[-1]["selector_regret"]) < float(history[0]["selector_regret"]),
        "reason_mix_synergy": float(reason_mix["Exp_mAP"]) > max(float(reason_global["Exp_mAP"]), float(reason_local["Exp_mAP"])) + 0.003,
        "reason_final_not_below_calalign": float(reason_final["Exp_mAP"]) >= float(reason_calalign["Exp_mAP"]),
        "annotation_residual_range": 0.005 <= float(reason_stats["reason_annotation_rms"]) <= 0.30,
        "factor_shuffle_hurts_12_labels": sum(
            1
            for normal, shuffled in zip(final_ap, shuffle_ap)
            if normal == normal and shuffled == shuffled and float(normal) > float(shuffled)
        ) >= 12,
        "meta_high_omega_4": sum(float(value) > 0.20 for value in latest["omega"]) >= 4,
        "meta_low_omega_4": sum(float(value) < 0.05 for value in latest["omega"]) >= 4,
        "meta_share_rate": 0.15 <= sum(float(value) > 0 for value in latest["omega"]) / max(len(latest["omega"]), 1) <= 0.60,
        "meta_positive_utility": any(float(value) > 0 for value in latest["utility_ema"]),
        "evidence_not_max_entropy": float(factor_stats["support_entropy_mean"]) < math.log(float(factor_stats["patch_count"]) + 1.0) - 1e-3,
        "support_null_nontrivial": 0.0 < float(factor_stats["support_null_mean"]) < 1.0,
        "counter_null_nontrivial": 0.0 < float(factor_stats["counter_null_mean"]) < 1.0,
        "counterfactual_all_actions": len(cf.get("covered_action_ids", [])) == 4,
        "counterfactual_12_factors": len(cf.get("covered_factor_ids", [])) >= 12,
        "selected_beats_control": float(cf.get("support_selected_effect_mean", 0.0)) > float(cf.get("support_control_effect_mean", 0.0)),
        "counter_direction": float(cf.get("counter_selected_effect_mean", 0.0)) > float(cf.get("counter_control_effect_mean", 0.0)),
    })
    try:
        remote_line = subprocess.check_output(
            ["git", "ls-remote", "github", f"refs/heads/{git_branch}"],
            cwd=root,
            text=True,
        ).strip()
        remote_head = remote_line.split()[0] if remote_line else ""
    except (OSError, subprocess.CalledProcessError):
        remote_head = ""
    checks["github_head_matches"] = remote_head == git_head
    passed = all(checks.values())
    report = {
        "artifact": "METER_OIA_V1_FULL_TRAIN_READY",
        "pass": passed,
        "HEAD": git_head,
        "github_head": remote_head,
        "branch": git_branch,
        "config_hash": config_hash,
        "schema_hash": schema_hash,
        "source_tree_hash": source_tree_hash,
        "checks": checks,
        "history": history,
        "internal_test_selected": True,
        "publication_eligible": False,
    }
    write_json(output_dir / "pilot_gate_report.json", report)
    ready_path = root / ".review" / "METER_OIA_V1_FULL_TRAIN_READY.json"
    if passed:
        write_json(ready_path, report)
    elif ready_path.exists():
        ready_path.unlink()
    return report


def validate_training_readiness(
    *,
    root: Path,
    config_path: Path,
    epochs: int,
    use_mock_dino: bool,
    git_head: str,
    git_branch: str,
    remote_head: str,
    clean_status: str,
    source_tree_hash: str,
) -> dict[str, Any]:
    """Validate the signed readiness payload, not merely its existence."""
    is_pilot = int(epochs) <= 3
    if use_mock_dino:
        phase = "pilot" if is_pilot else "full"
        raise RuntimeError(f"{phase} training cannot use mock DINO")
    readiness_name = (
        "METER_OIA_V1_PRE_PILOT_READY.json"
        if is_pilot
        else "METER_OIA_V1_FULL_TRAIN_READY.json"
    )
    expected_artifact = (
        "METER_OIA_V1_PRE_PILOT_READY"
        if is_pilot
        else "METER_OIA_V1_FULL_TRAIN_READY"
    )
    ready_path = root / ".review" / readiness_name
    if not ready_path.exists():
        raise RuntimeError(f"{readiness_name} is required before training")
    payload = json.loads(ready_path.read_text(encoding="utf-8-sig"))
    expected_config_hash = file_hash(config_path)
    expected_schema_hash = combined_file_hash(
        root / "configs/meter_factor_schema.yaml",
        root / "configs/meter_grounding_schema.yaml",
    )
    checks = {
        "artifact": payload.get("artifact") == expected_artifact,
        "HEAD": payload.get("HEAD") == git_head,
        "config_hash": payload.get("config_hash") == expected_config_hash,
        "schema_hash": payload.get("schema_hash") == expected_schema_hash,
        "source_tree_hash": payload.get("source_tree_hash") == source_tree_hash,
        "clean_worktree": clean_status == "",
        "branch": payload.get("branch") == git_branch,
        "remote_HEAD": remote_head == git_head,
    }
    if is_pilot:
        checks.update({
            "unresolved": payload.get("unresolved") == [],
            "real_dino": bool(payload.get("real_dino", {}).get("pass")),
        })
    else:
        gate_checks = payload.get("checks", {})
        checks.update({
            "pass=true": payload.get("pass") is True,
            "required_check_keys": FULL_READY_REQUIRED_CHECKS.issubset(gate_checks),
            "all_checks": FULL_READY_REQUIRED_CHECKS.issubset(gate_checks)
            and all(gate_checks[key] is True for key in FULL_READY_REQUIRED_CHECKS),
            "gate_github_head": payload.get("github_head") == git_head,
        })
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"{readiness_name} validation failed: {', '.join(failed)}"
        )
    return payload


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    cfg = load_meter_config(config_path)
    if args.batch_size:
        cfg["training"]["batch_size"] = int(args.batch_size)
    grad_accum = int(args.gradient_accumulation_steps or cfg["training"].get("gradient_accumulation_steps", 1))
    cfg["training"]["gradient_accumulation_steps"] = grad_accum
    if grad_accum < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    root = Path(args.worktree_root or ".").resolve()
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip()
        clean_status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
        remote_line = subprocess.check_output(
            ["git", "ls-remote", "github", f"refs/heads/{git_branch}"],
            cwd=root,
            text=True,
        ).strip()
        remote_head = remote_line.split()[0] if remote_line else ""
    except (OSError, subprocess.CalledProcessError):
        git_head, git_branch, clean_status, remote_head = (
            "unknown", "unknown", "git-state-unavailable", ""
        )
    current_source_tree_hash = python_source_tree_hash(root)
    validate_training_readiness(
        root=root,
        config_path=config_path.resolve(),
        epochs=int(args.epochs),
        use_mock_dino=bool(args.use_mock_dino),
        git_head=git_head,
        git_branch=git_branch,
        remote_head=remote_head,
        clean_status=clean_status,
        source_tree_hash=current_source_tree_hash,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg["training"].get("tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(cfg["training"].get("tf32", True))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = meter_image_transform()
    grounding_index = METERGroundingIndex(cfg["data"]["bdd100k_root"], schema_path="configs/meter_factor_schema.yaml")
    dataset = METERDataset(
        data_root=cfg["data"]["data_root"], raw_root=cfg["data"]["raw_root"], split="train",
        transform=transform, grounding_index=grounding_index, include_grounding=True,
    )
    split = fixed_meter_split_indices([s.file_name for s in dataset.base.samples], audit_fraction=cfg["splits"]["audit_fraction"], calib_fraction=cfg["splits"]["calib_fraction"], seed=cfg["splits"]["seed"])
    main_indices = split["main"][: args.max_train_samples] if args.max_train_samples else split["main"]
    audit_indices = split["audit"][: args.max_audit_samples] if args.max_audit_samples else split["audit"]
    calib_indices = split["calib"][: args.max_calib_samples] if args.max_calib_samples else split["calib"]
    model = METEROIAModel(
        dim=cfg["model"]["dim"], action_dim=cfg["model"]["action_dim"], reason_dim=cfg["model"]["reason_dim"],
        selected_layers=tuple(cfg["backbone"]["selected_layers"]), pretrained_weights=cfg["backbone"]["pretrained_weights"],
        use_mock_dino=args.use_mock_dino, factor_rank=cfg["model"].get("factor_rank", 16),
    ).to(device)
    model.foundation.dino.eval()
    optimizer = _make_optimizer(model, cfg)
    micro_batches_per_epoch = max(1, math.ceil(len(main_indices) / int(cfg["training"].get("batch_size", 6))))
    updates_per_epoch = max(1, math.ceil(micro_batches_per_epoch / grad_accum))
    total_updates = max(1, int(args.epochs) * updates_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, int(total_updates * float(cfg["training"].get("warmup_fraction", 0.05)))))
        * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(step / max(total_updates, 1), 1.0)))),
    )
    audit_loader = _loader(dataset, audit_indices, cfg, shuffle=False)
    calib_loader = _loader(dataset, calib_indices, cfg, shuffle=False)
    meta = METERMetaUtility(
        factors=cfg["model"]["reason_dim"], virtual_lr=cfg["meta"]["virtual_lr"],
        ema_old_weight=cfg["meta"]["ema_old_weight"], ema_new_weight=cfg["meta"]["ema_new_weight"],
        lower=cfg["meta"]["utility_lower"], upper=cfg["meta"]["utility_upper"],
    )
    baseline_hash = str(cfg["audit"]["require_source_sha"])
    config_hash = file_hash(config_path)
    schema_hash = combined_file_hash(
        root / "configs/meter_factor_schema.yaml",
        root / "configs/meter_grounding_schema.yaml",
    )
    source_hash = git_head
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(output_dir / "config_resolved.yaml.json", cfg)
    write_json(output_dir / "source_fingerprint.json", {"git_head": git_head, "branch": git_branch, "base_sha": baseline_hash, "source_hash": source_hash, "config_hash": config_hash, "schema_hash": schema_hash})
    owner_manifest = {
        "owners": [],
        "dino_in_optimizer": False,
        "parameter_overlap": False,
    }
    seen_parameter_ids: set[int] = set()
    for group in optimizer.param_groups:
        owner = str(group.get("name", "unclassified"))
        group_parameters = list(group.get("params", []))
        overlap = any(id(parameter) in seen_parameter_ids for parameter in group_parameters)
        seen_parameter_ids.update(id(parameter) for parameter in group_parameters)
        owner_manifest["owners"].append({
            "name": owner,
            "parameter_count": len(group_parameters),
            "trainable_numel": int(sum(parameter.numel() for parameter in group_parameters)),
            "learning_rate": float(group["lr"]),
            "parameter_overlap": bool(overlap),
        })
    owner_manifest["parameter_overlap"] = any(bool(item["parameter_overlap"]) for item in owner_manifest["owners"])
    write_json(output_dir / "owner_manifest.json", owner_manifest)
    runtime_profile_path = Path(".review/meter_oia_v1_real_profile/runtime_profile.json")
    runtime_profile = {}
    if runtime_profile_path.exists():
        try:
            runtime_profile = json.loads(runtime_profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            runtime_profile = {"available": False, "reason": "runtime_profile_unreadable"}
    if not runtime_profile:
        runtime_profile = {"available": False, "reason": "runtime_profile_not_found"}
    write_json(output_dir / "runtime_profile.json", runtime_profile)
    write_json(output_dir / "run_manifest.json", {
        "experiment": cfg["experiment"], "command_line": " ".join(os.sys.argv), "git_head": git_head, "branch": git_branch, "github_main_baseline_head": baseline_hash, "source_hash": source_hash,
        "config_hash": config_hash, "schema_hash": schema_hash, "data_root": cfg["data"]["data_root"],
        "raw_root": cfg["data"]["raw_root"], "test_only": True, "best_selection_split": "test",
        "feature_cache_enabled": False, "token_compression": "none", "internal_test_selected": True,
        "publication_eligible": False, "split_sizes": {k: len(v) for k, v in split.items()},
        "split_manifest": meter_split_manifest([s.file_name for s in dataset.base.samples], split),
        "batch_size": int(cfg["training"].get("batch_size", 6)), "gradient_accumulation_steps": grad_accum,
        "effective_batch": int(cfg["training"].get("batch_size", 6)) * grad_accum,
        "selected_layers": list(cfg["backbone"]["selected_layers"]), "pretrained_weights": cfg["backbone"]["pretrained_weights"],
        "optimizer_owners": [group.get("name") for group in optimizer.param_groups],
        "loss_weights": cfg["loss_weights"], "foreground_only": True, "no_feature_cache": True,
        "runtime_profile_path": str(output_dir / "runtime_profile.json"),
        "owner_manifest_path": str(output_dir / "owner_manifest.json"),
        "optimizer_owner_contract": owner_manifest,
    })
    global_step = 0
    best_joint = float("-inf")
    best_branch_metrics: dict[str, float] = {}
    pu_lambda = torch.zeros(model.reason_decoder.reason_dim, device=device)
    pu_audit_state: dict[str, Any] = {"active_labels": [], "labels": []}
    start_epoch = 0
    resume_micro_step = 0
    calibration = METERCalibrationResult(
        theta=torch.zeros(model.foundation.action_dim + model.foundation.reason_dim),
        temperature=torch.ones(model.foundation.action_dim + model.foundation.reason_dim),
        model_state_hash_before=state_hash(model),
        model_state_hash_after=state_hash(model),
        fit_split="train_calib",
        representation_updated=False,
        strategy="global_raw",
    )
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, expected_config_hash=config_hash, expected_source_hash=source_hash, expected_schema_hash=schema_hash)
        global_step = int(payload.get("optimizer_step", 0))
        resume_micro_step = int(payload.get("micro_step", 0))
        start_epoch = int(payload.get("epoch", -1)) + (0 if resume_micro_step > 0 else 1)
        meta_state = payload.get("meta_state", {})
        if meta_state.get("omega") is not None:
            meta.omega = torch.as_tensor(meta_state["omega"], dtype=meta.omega.dtype).clone()
        if meta_state.get("utility_ema") is not None:
            meta.utility_ema = torch.as_tensor(meta_state["utility_ema"], dtype=meta.utility_ema.dtype).clone()
        meta.cursor = int(meta_state.get("cursor", meta.cursor))
        pu_state = payload.get("pu_state", {})
        if pu_state.get("lambda") is not None:
            pu_lambda = torch.as_tensor(pu_state["lambda"], device=device, dtype=torch.float32)
        calibration_payload = payload.get("calibration", {})
        if calibration_payload.get("theta") is not None:
            calibration = METERCalibrationResult(
                theta=torch.as_tensor(calibration_payload["theta"]),
                temperature=(
                    None
                    if calibration_payload.get("temperature") is None
                    else torch.as_tensor(calibration_payload["temperature"])
                ),
                strategy=str(calibration_payload.get("strategy", "per_label")),
                model_state_hash_before="",
                model_state_hash_after="",
                fit_split="train_calib",
                representation_updated=False,
                accepted=bool(calibration_payload.get("accepted", True)),
                fallback_reason=str(calibration_payload.get("fallback_reason", "")),
                train_calib_raw_joint=calibration_payload.get("train_calib_raw_joint"),
                train_calib_deploy_joint=calibration_payload.get("train_calib_deploy_joint"),
                map_max_abs_delta=calibration_payload.get("map_max_abs_delta"),
                threshold_rms_ratio=calibration_payload.get("threshold_rms_ratio"),
            )
        best_joint = float(payload.get("meta_state", {}).get("best_joint", float("-inf")))
        best_branch_metrics = {str(key): float(value) for key, value in payload.get("meta_state", {}).get("best_branch_metrics", {}).items()}
    with torch.no_grad():
        model.meta_share_weight.copy_(meta.omega.to(device=model.meta_share_weight.device, dtype=model.meta_share_weight.dtype))
    amp_enabled = bool(cfg["training"].get("bf16", False)) and device.type == "cuda"
    amp_context = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled) if device.type in {"cuda", "cpu"} else nullcontext()
    meta_every_updates = int(cfg["meta"].get("pilot_every_updates", 40) if int(args.epochs) <= 3 else cfg["meta"].get("full_every_updates", 100))
    print_every = int(cfg["runtime"].get("print_every_optimizer_updates", 50))
    owner_step_counts: dict[str, int] = {}
    owner_zero_gradient_counts: dict[str, int] = {}
    pilot_history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        model.foundation.dino.eval()
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        epoch_selector_regret: list[float] = []
        train_loader = _loader(
            dataset,
            main_indices,
            cfg,
            shuffle=True,
            seed=int(cfg["splits"].get("seed", 20260728)) + epoch,
        )
        iterator = iter(train_loader)
        epoch_resume_micro = resume_micro_step if epoch == start_epoch else 0
        for _ in range(epoch_resume_micro):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError("Resume micro_step exceeds the deterministic epoch loader") from exc
        for micro in range(epoch_resume_micro, micro_batches_per_epoch):
            data_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            data_time = time.perf_counter() - data_started
            images = batch["image"].to(device, non_blocking=True)
            is_update = ((micro + 1) % grad_accum == 0) or micro == micro_batches_per_epoch - 1
            collect_timing = is_update and (global_step + 1) % max(print_every, 1) == 0
            with amp_context():
                if collect_timing and device.type == "cuda":
                    torch.cuda.synchronize(device)
                dino_started = time.perf_counter()
                field = model.encode_images(images)
                if collect_timing and device.type == "cuda":
                    torch.cuda.synchronize(device)
                dino_time = time.perf_counter() - dino_started
                progress = min(global_step / max(total_updates, 1), 1.0)
                output = model.decode_from_field(
                    field,
                    progress=progress,
                    collect_timing=collect_timing,
                )
                counterfactual = None
                if is_update and (global_step + 1) % int(cfg["counterfactual"].get("every_optimizer_updates", 8)) == 0:
                    counterfactual = _counterfactual_event(
                        model,
                        field,
                        output,
                        progress,
                        action_target=batch["action"].to(device),
                        selected_mass=float(cfg["counterfactual"].get("selected_mass", 0.60)),
                        max_patches=int(cfg["counterfactual"].get("max_patches", 28)),
                        minimum_patches=int(cfg["counterfactual"].get("minimum_patches", 4)),
                        sample_index=global_step % max(int(images.shape[0]), 1),
                    )
                    append_jsonl(output_dir / "counterfactual_events.jsonl", {"epoch": epoch, "step": global_step, **{key: (float(value.detach().cpu()) if isinstance(value, Tensor) and value.ndim == 0 else (value.detach().cpu().tolist() if isinstance(value, Tensor) else value)) for key, value in counterfactual.items() if key != "total"}})
                total, parts, diagnostics = _losses(
                    model,
                    output,
                    batch,
                    cfg,
                    progress,
                    device,
                    pu_lambda=pu_lambda,
                    counterfactual=counterfactual,
                    counterfactual_event_scale=float(grad_accum if counterfactual is not None else 1),
                )
            if collect_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            backward_started = time.perf_counter()
            (total / grad_accum).backward()
            if collect_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            backward_time = time.perf_counter() - backward_started
            if not is_update:
                continue
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 1.0))).detach().cpu())
            owner_grad_norms: dict[str, float] = {}
            owner_delta_estimates: dict[str, float] = {}
            owner_has_gradient: dict[str, bool] = {}
            for group in optimizer.param_groups:
                owner = str(group.get("name", "unclassified"))
                squared = sum((parameter.grad.detach().float().square().sum() for parameter in group["params"] if parameter.grad is not None), torch.zeros((), device=device))
                owner_grad_norms[owner] = (
                    owner_grad_norms.get(owner, 0.0) ** 2 + float(squared.detach().cpu())
                ) ** 0.5
                owner_delta_estimates[owner] = float(owner_grad_norms[owner] * float(group["lr"]))
                owner_has_gradient[owner] = owner_has_gradient.get(owner, False) or any(
                    parameter.grad is not None and bool(parameter.grad.detach().ne(0).any())
                    for parameter in group["params"]
                )
            for owner, has_gradient in owner_has_gradient.items():
                owner_step_counts[owner] = owner_step_counts.get(owner, 0) + 1
                if not has_gradient:
                    owner_zero_gradient_counts[owner] = owner_zero_gradient_counts.get(owner, 0) + 1
            parameter_before: dict[int, Tensor] = {}
            if collect_timing:
                parameter_before = {
                    id(parameter): parameter.detach().clone()
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                }
            if collect_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            optimizer_started = time.perf_counter()
            optimizer.step()
            if collect_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            optimizer_time = time.perf_counter() - optimizer_started
            owner_parameter_delta: dict[str, float] = {}
            if collect_timing:
                for group in optimizer.param_groups:
                    owner = str(group.get("name", "unclassified"))
                    squared_delta = sum(
                        (
                            parameter.detach().float() - parameter_before[id(parameter)].float()
                        ).square().sum()
                        for parameter in group["params"]
                    )
                    owner_parameter_delta[owner] = (
                        owner_parameter_delta.get(owner, 0.0) ** 2
                        + float(squared_delta.detach().cpu())
                    ) ** 0.5
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if global_step % max(meta_every_updates, 1) == 0 and audit_indices:
                audit_batch = next(iter(audit_loader))
                audit_field = model.encode_images(audit_batch["image"].to(device, non_blocking=True))
                parameter_map = {"down": model.signed_factors.meta_adapters.down, "up": model.signed_factors.meta_adapters.up}
                audit_action = audit_batch["action"].to(device)
                train_action = batch["action"].to(device)
                factor_count = min(int(cfg["meta"]["factors_per_event"]), model.reason_decoder.reason_dim)
                factor_ids = tuple((meta.cursor + offset) % model.reason_decoder.reason_dim for offset in range(factor_count))
                def action_loss_fn(params: dict[str, Tensor]) -> Tensor:
                    virtual = model.decode_from_field(field, progress=progress, factor_parameter_override=params)
                    return meter_action_loss(virtual, train_action, cfg["loss_weights"])["total"]
                def reason_loss_fn(params: dict[str, Tensor]) -> Tensor:
                    virtual = model.decode_from_field(field, progress=progress, factor_parameter_override=params, meta_share_weight_override=torch.ones(model.reason_decoder.reason_dim, device=device))
                    return meter_reason_loss(
                        virtual,
                        batch["reason"].to(device),
                        virtual["factor_reliability"],
                        cfg["loss_weights"],
                        observability=_move_grounding(batch, device)["factor_observability"],
                    )["total"]
                def heldout_action(params: dict[str, Tensor]) -> Tensor:
                    virtual = model.decode_from_field(audit_field, progress=progress, factor_parameter_override=params)
                    return meter_action_loss(virtual, audit_action, cfg["loss_weights"])["total"]
                event = meta.event(parameter_map, factor_ids=factor_ids, dino_calls=1, action_loss_fn=action_loss_fn, reason_loss_fn=reason_loss_fn, audit_action_loss_fn=heldout_action)
                with torch.no_grad():
                    model.meta_share_weight.copy_(meta.omega.to(device=model.meta_share_weight.device, dtype=model.meta_share_weight.dtype))
                append_jsonl(output_dir / "meta_utility.jsonl", {
                    "epoch": epoch,
                    "step": global_step,
                    "factor_ids": event.factor_ids,
                    "action_only_loss": event.action_only_loss,
                    "action_reason_loss": event.action_reason_loss,
                    "relative_utility": event.relative_utility.detach().cpu().tolist(),
                    "relative_utility_by_factor": {
                        str(factor_id): float(value)
                        for factor_id, value in zip(
                            event.factor_ids,
                            event.relative_utility.detach().cpu().reshape(-1).tolist(),
                        )
                    },
                    "omega_before": event.omega_before.tolist(),
                    "omega_after": event.omega_after.tolist(),
                    "action_grad_norm": event.action_grad_norm,
                    "reason_grad_norm": event.reason_grad_norm,
                    "candidate_delta_norm": event.candidate_delta_norm,
                    "wall_time_sec": event.wall_time_sec,
                    "dino_calls": event.dino_calls,
                    "train_audit_only": True,
                    "test_used": False,
                })
                meta.cursor = (meta.cursor + len(factor_ids)) % model.reason_decoder.reason_dim
            append_jsonl(output_dir / "loss_components.jsonl", {
                "epoch": epoch,
                "step": global_step,
                "loss_total": float(total.detach().cpu()),
                "loss_action": float(parts["action"].detach().cpu()),
                "loss_reason": float(parts["reason"].detach().cpu()),
                "loss_grounding": float(parts["grounding"].detach().cpu()),
                "loss_pu": float(parts["pu"].detach().cpu()),
                "loss_counterfactual": float(parts["counterfactual"].detach().cpu()),
                "grad_norm": grad_norm,
                "effective_batch": int(cfg["training"].get("batch_size", 6)) * grad_accum,
                "pu_active_labels": [int(x) for x in torch.where(pu_lambda > 0)[0].detach().cpu().tolist()],
                "counterfactual_valid_count": int(diagnostics["counterfactual"].get("valid_count", 0)),
                "counterfactual_selected_patch_count": int(diagnostics["counterfactual"].get("selected_patch_count", 0)),
                "lr_by_owner": {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups},
                "owner_grad_norms": owner_grad_norms,
                "owner_delta_estimates": owner_delta_estimates,
                "owner_parameter_delta": owner_parameter_delta,
                "owner_optimizer_step_count": dict(owner_step_counts),
                "owner_zero_gradient_rate": {
                    owner: owner_zero_gradient_counts.get(owner, 0) / max(count, 1)
                    for owner, count in owner_step_counts.items()
                },
                "runtime": {
                    "data_time": data_time,
                    "dino_time": dino_time if collect_timing else None,
                    **output.get("runtime_timing", {}),
                    "backward_time": backward_time if collect_timing else None,
                    "optimizer_time": optimizer_time if collect_timing else None,
                    "samples_per_sec": (
                        float(images.shape[0])
                        / max(
                            data_time
                            + dino_time
                            + sum(output.get("runtime_timing", {}).values())
                            + backward_time
                            + optimizer_time,
                            1e-9,
                        )
                        if collect_timing
                        else None
                    ),
                    "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0,
                    "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0,
                    "dino_call_count": model.foundation.ordinary_dino_calls,
                },
                "loss_breakdown": {
                    "action": {key: float(value.detach().cpu()) for key, value in diagnostics["action"].items() if isinstance(value, Tensor)},
                    "reason": {key: float(value.detach().cpu()) for key, value in diagnostics["reason"].items() if isinstance(value, Tensor)},
                    "grounding": {key: float(value.detach().cpu()) for key, value in diagnostics["grounding"].items() if isinstance(value, Tensor)},
                    "counterfactual": {key: float(value.detach().cpu()) for key, value in diagnostics["counterfactual"].items() if isinstance(value, Tensor) and value.ndim == 0},
                },
            })
            epoch_selector_regret.append(float(diagnostics["action"]["selector_regret"].detach().cpu()))
            if global_step % max(print_every, 1) == 0:
                print(json.dumps({"meter_batch": {"epoch": epoch, "step": global_step, "loss": float(total.detach().cpu()), "action": float(parts["action"].detach().cpu()), "reason": float(parts["reason"].detach().cpu()), "grounding": float(parts["grounding"].detach().cpu()), "pu": float(parts["pu"].detach().cpu()), "grad_norm": grad_norm}}, sort_keys=True), flush=True)
                save_checkpoint(
                    output_dir / "checkpoint_latest.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    micro_step=micro + 1,
                    optimizer_step=global_step,
                    runtime_profile={
                        "batch_size": cfg["training"].get("batch_size"),
                        "gradient_accumulation_steps": grad_accum,
                        "effective_batch": int(cfg["training"].get("batch_size", 6)) * grad_accum,
                        "runtime_profile": runtime_profile,
                    },
                    meta_state={
                        "omega": meta.omega,
                        "utility_ema": meta.utility_ema,
                        "cursor": meta.cursor,
                        "best_joint": best_joint,
                        "best_branch_metrics": best_branch_metrics,
                    },
                    pu_state={
                        "lambda": pu_lambda.detach().cpu(),
                        "audit": pu_audit_state,
                    },
                    calibration={
                        "theta": calibration.theta.detach().cpu(),
                        "temperature": None if calibration.temperature is None else calibration.temperature.detach().cpu(),
                        "strategy": calibration.strategy,
                        "accepted": calibration.accepted,
                        "fallback_reason": calibration.fallback_reason,
                        "train_calib_raw_joint": calibration.train_calib_raw_joint,
                        "train_calib_deploy_joint": calibration.train_calib_deploy_joint,
                    },
                    config_hash=config_hash,
                    source_hash=source_hash,
                    schema_hash=schema_hash,
                )
        resume_micro_step = 0
        audit_outputs = collect_outputs(model, audit_loader, device, progress=1.0)
        audit_mechanism = audit_outputs.get("mechanism", {})
        pu_audit_state = meter_hidden_positive_audit(
            torch.sigmoid(audit_outputs["reason"]["global"]),
            torch.sigmoid(
                audit_mechanism.get("factor_support_score", torch.zeros_like(audit_outputs["reason"]["global"]))
                - audit_mechanism.get("factor_counter_score", torch.zeros_like(audit_outputs["reason"]["global"]))
            ),
            audit_outputs["labels_reason"],
            hidden_fraction=float(cfg["pu"].get("hidden_positive_fraction", 0.30)),
            min_positive_count=int(cfg["pu"].get("min_positive_count", 20)),
            seed=int(cfg["splits"].get("seed", 20260728)) + epoch,
        )
        if epoch >= 0:
            pu_lambda = torch.as_tensor(pu_audit_state["lambda"], device=device, dtype=torch.float32).clamp_min(0.0).clamp_max(float(cfg["pu"].get("max_lambda", 0.15)))
        append_jsonl(output_dir / "pu_audit.jsonl", {"epoch": epoch, **pu_audit_state, "active_lambda": pu_lambda.detach().cpu().tolist()})
        calibration = _fit_calibration(
            model,
            calib_loader,
            device,
            progress=1.0,
            fallback_on_deploy_degradation=bool(cfg["posthoc_calibration"].get("fallback_on_deploy_degradation", True)),
        )
        test_result = evaluate_test(model, cfg, device, args, progress=1.0, calibration=calibration)
        raw = test_result["summary"]["metrics_raw"]
        deploy = test_result["summary"]["metrics_deploy"]
        joint = float(deploy.get("deploy_joint", 0.0))
        append_jsonl(output_dir / "metrics_summary.jsonl", {"epoch": epoch, "metrics_raw": raw, "metrics_deploy": deploy, "runtime_sec": time.time() - epoch_start, "dino_calls": model.foundation.ordinary_dino_calls})
        append_jsonl(output_dir / "mechanism_stats.jsonl", {"epoch": epoch, "branch_metrics": test_result["branches"], "factor_stats": test_result["factor_stats"], "selector_stats": test_result["selector_stats"], "reason_view_stats": test_result["reason_view_stats"], "factor_reliability_mean": float(test_result["factor_reliability_mean"]), "factor_support_mean": float(test_result["factor_support_mean"]), "pu_active_labels": pu_audit_state["active_labels"]})
        pilot_history.append({
            "epoch": epoch,
            "finite": all(
                value == value and math.isfinite(float(value))
                for value in (
                    deploy.get("Act_mF1", float("nan")),
                    deploy.get("Exp_mF1", float("nan")),
                    deploy.get("Act_mAP", float("nan")),
                    deploy.get("Exp_mAP", float("nan")),
                )
            ),
            "branches": test_result["branches"],
            "factor_stats": test_result["factor_stats"],
            "selector_stats": test_result["selector_stats"],
            "reason_view_stats": test_result["reason_view_stats"],
            "counterfactual": test_result["counterfactual_stats"],
            "selector_regret": sum(epoch_selector_regret) / max(len(epoch_selector_regret), 1),
            "omega": meta.omega.tolist(),
            "utility_ema": meta.utility_ema.tolist(),
            "owner_step_counts": dict(owner_step_counts),
            "owner_zero_gradient_rate": {
                owner: owner_zero_gradient_counts.get(owner, 0) / max(count, 1)
                for owner, count in owner_step_counts.items()
            },
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0,
        })
        diagnostics = {
            "factor_stats": test_result["factor_stats"],
            "evidence_maps_stats": test_result["evidence_maps_stats"],
            "selector_stats": test_result["selector_stats"],
            "reason_view_stats": test_result["reason_view_stats"],
            "meta_stats": {"omega": meta.omega.tolist(), "utility_ema": meta.utility_ema.tolist(), "cursor": meta.cursor},
            "pu_stats": pu_audit_state,
            "counterfactual.json": test_result["counterfactual_stats"],
            "per_action.json": {"raw": {k: v for k, v in raw.items() if k.startswith("Act_")}, "deploy": {k: v for k, v in deploy.items() if k.startswith("Act_")}},
            "per_reason.json": {"raw": {k: v for k, v in raw.items() if k.startswith("Exp_")}, "deploy": {k: v for k, v in deploy.items() if k.startswith("Exp_")}},
            "failure_cases.jsonl": test_result["failure_cases"],
            "evidence_cases.jsonl": test_result["evidence_cases"],
            "calibration.json": {
                "theta": calibration.theta.tolist(),
                "temperature": None if calibration.temperature is None else calibration.temperature.tolist(),
                "strategy": calibration.strategy,
                "fit_split": calibration.fit_split,
                "state_hash_before": calibration.model_state_hash_before,
                "state_hash_after": calibration.model_state_hash_after,
                "accepted": calibration.accepted,
                "fallback_reason": calibration.fallback_reason,
                "train_calib_raw_joint": calibration.train_calib_raw_joint,
                "train_calib_deploy_joint": calibration.train_calib_deploy_joint,
                "map_max_abs_delta": calibration.map_max_abs_delta,
                "threshold_rms_ratio": calibration.threshold_rms_ratio,
            },
        }
        save_epoch_artifacts(output_dir, epoch, metrics_raw=raw, metrics_deploy=deploy, branch_metrics=test_result["branches"], logits=test_result["logits"], labels=test_result["labels"], diagnostics=diagnostics, file_names=test_result["file_names"])
        checkpoint_runtime = {"batch_size": cfg["training"].get("batch_size"), "gradient_accumulation_steps": grad_accum, "effective_batch": int(cfg["training"].get("batch_size", 6)) * grad_accum, "runtime_profile": runtime_profile}
        checkpoint_meta = {"omega": meta.omega, "utility_ema": meta.utility_ema, "cursor": meta.cursor, "best_joint": best_joint, "best_branch_metrics": best_branch_metrics}
        calibration_payload = {
            "theta": calibration.theta.detach().cpu(),
            "temperature": None if calibration.temperature is None else calibration.temperature.detach().cpu(),
            "strategy": calibration.strategy,
            "accepted": calibration.accepted,
            "fallback_reason": calibration.fallback_reason,
            "train_calib_raw_joint": calibration.train_calib_raw_joint,
            "train_calib_deploy_joint": calibration.train_calib_deploy_joint,
            "map_max_abs_delta": calibration.map_max_abs_delta,
            "threshold_rms_ratio": calibration.threshold_rms_ratio,
        }
        save_checkpoint(output_dir / "checkpoint_latest.pth", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, micro_step=0, optimizer_step=global_step, runtime_profile=checkpoint_runtime, meta_state=checkpoint_meta, pu_state={"lambda": pu_lambda.detach().cpu(), "audit": pu_audit_state}, calibration=calibration_payload, config_hash=config_hash, source_hash=source_hash, schema_hash=schema_hash)
        if joint > best_joint:
            best_joint = joint
            checkpoint_meta["best_joint"] = best_joint
            save_checkpoint(output_dir / "checkpoint_best_test_deploy_joint.pth", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, micro_step=0, optimizer_step=global_step, runtime_profile=checkpoint_runtime, meta_state=checkpoint_meta, pu_state={"lambda": pu_lambda.detach().cpu(), "audit": pu_audit_state}, calibration=calibration_payload, config_hash=config_hash, source_hash=source_hash, schema_hash=schema_hash)
        branch_checkpoint_metrics = {
            "action_mf1": float(deploy.get("Act_mF1", float("-inf"))),
            "action_map": float(deploy.get("Act_mAP", float("-inf"))),
            "reason_mf1": float(deploy.get("Exp_mF1", float("-inf"))),
            "reason_map": float(deploy.get("Exp_mAP", float("-inf"))),
            "visual_action_mf1": float(test_result["branches"].get("action_visual", {}).get("Act_mF1", float("-inf"))),
            "semantic_action_mf1": float(test_result["branches"].get("action_semantic", {}).get("Act_mF1", float("-inf"))),
        }
        best_checkpoint_names = {
            "action_mf1": "checkpoint_best_test_action_mf1.pth",
            "action_map": "checkpoint_best_test_action_map.pth",
            "reason_mf1": "checkpoint_best_test_exp_mf1.pth",
            "reason_map": "checkpoint_best_test_exp_map.pth",
            "visual_action_mf1": "checkpoint_best_test_visual_action.pth",
            "semantic_action_mf1": "checkpoint_best_test_semantic_action.pth",
        }
        for metric_name, metric_value in branch_checkpoint_metrics.items():
            if metric_value > float(best_branch_metrics.get(metric_name, float("-inf"))):
                best_branch_metrics[metric_name] = metric_value
                checkpoint_meta["best_branch_metrics"] = best_branch_metrics
                save_checkpoint(output_dir / best_checkpoint_names[metric_name], model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, micro_step=0, optimizer_step=global_step, runtime_profile=checkpoint_runtime, meta_state=checkpoint_meta, pu_state={"lambda": pu_lambda.detach().cpu(), "audit": pu_audit_state}, calibration=calibration_payload, config_hash=config_hash, source_hash=source_hash, schema_hash=schema_hash)
        print(json.dumps({"epoch": epoch, "test": {"Act_mF1": deploy.get("Act_mF1"), "Act_oF1": deploy.get("Act_oF1"), "Exp_mF1": deploy.get("Exp_mF1"), "Exp_oF1": deploy.get("Exp_oF1"), "deploy_joint": joint}, "best_joint": best_joint}, sort_keys=True), flush=True)
    if int(args.epochs) == 3:
        _write_full_train_ready_if_eligible(
            root=Path(args.worktree_root or "."),
            output_dir=output_dir,
            args=args,
            git_head=git_head,
            git_branch=git_branch,
            config_hash=config_hash,
            schema_hash=schema_hash,
            source_tree_hash=current_source_tree_hash,
            history=pilot_history,
        )


@torch.no_grad()
def _test_counterfactual_diagnostic(model: METEROIAModel, dataset: METERDataset, device: torch.device, *, batch_size: int, num_workers: int, max_samples: int, progress: float, selected_mass: float, max_patches: int, minimum_patches: int) -> dict[str, Any]:
    loader = DataLoader(Subset(dataset, list(range(min(len(dataset), max_samples)))), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=2)
    records: list[dict[str, Any]] = []
    seen = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        field = model.encode_images(images)
        output = model.decode_from_field(field, progress=progress)
        event = _counterfactual_event(
            model,
            field,
            output,
            progress,
            action_target=batch["action"].to(device),
            selected_mass=selected_mass,
            max_patches=max_patches,
            minimum_patches=minimum_patches,
            sample_index=seen % max(int(images.shape[0]), 1),
        )
        records.append(event)
        seen += int(images.shape[0])
        if seen >= max_samples:
            break
    if not records:
        return {"available": False, "reason": "no_samples", "valid_count": 0}
    def mean_value(key: str) -> float:
        values = [record[key].detach().float().mean().item() if isinstance(record.get(key), Tensor) else float(record.get(key, 0.0)) for record in records]
        return float(sum(values) / max(len(values), 1))
    return {
        "available": True,
        "valid_count": int(sum(int(record.get("valid_count", 0)) for record in records)),
        "selected_patch_count": int(max(int(record.get("selected_patch_count", 0)) for record in records)),
        "support_selected_effect_mean": mean_value("support_selected_effect"),
        "support_control_effect_mean": mean_value("support_control_effect"),
        "counter_selected_effect_mean": mean_value("counter_selected_effect"),
        "counter_control_effect_mean": mean_value("counter_control_effect"),
        "target_action_effect_mean": mean_value("target_action_effect"),
        "control_action_effect_mean": mean_value("control_action_effect"),
        "wrong_action_effect_mean": mean_value("wrong_action_effect"),
        "support_control_overlap": int(sum(int(record.get("selected_control_overlap", 0)) for record in records)),
        "counter_control_overlap": int(sum(int(record.get("counter_control_overlap", 0)) for record in records)),
        "covered_action_ids": sorted({
            int(record["target_action_id"])
            for record in records
            if int(record.get("valid_count", 0)) > 0 and "target_action_id" in record
        }),
        "covered_factor_ids": sorted({
            int(record["chosen_factor_id"])
            for record in records
            if int(record.get("valid_count", 0)) > 0 and "chosen_factor_id" in record
        }),
    }


def evaluate_test(model: METEROIAModel, cfg: dict[str, Any], device: torch.device, args: argparse.Namespace, *, progress: float, calibration: METERCalibrationResult) -> dict[str, Any]:
    dataset = METERDataset(data_root=cfg["data"]["data_root"], raw_root=cfg["data"]["raw_root"], split="test", transform=meter_image_transform(), grounding_index=None, include_grounding=False)
    indices = list(range(len(dataset)))
    if args.max_test_samples:
        indices = indices[: args.max_test_samples]
    loader = DataLoader(Subset(dataset, indices), batch_size=int(cfg["training"].get("batch_size", 6)), shuffle=False, num_workers=int(cfg["data"].get("num_workers", 4)), pin_memory=True, persistent_workers=int(cfg["data"].get("num_workers", 4)) > 0, prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)))
    collected = collect_outputs(model, loader, device, progress=progress)
    summary = metrics_summary(collected, calibration)
    stats = mechanism_stats_from_collected(collected)
    action_dim = collected["action"]["final"].shape[1]
    action_calibration = METERCalibrationResult(
        theta=calibration.theta[:action_dim],
        temperature=None if calibration.temperature is None else calibration.temperature[:action_dim],
        model_state_hash_before=calibration.model_state_hash_before,
        model_state_hash_after=calibration.model_state_hash_after,
        fit_split="train_calib",
        representation_updated=False,
    )
    reason_calibration = METERCalibrationResult(
        theta=calibration.theta[action_dim:],
        temperature=None if calibration.temperature is None else calibration.temperature[action_dim:],
        model_state_hash_before=calibration.model_state_hash_before,
        model_state_hash_after=calibration.model_state_hash_after,
        fit_split="train_calib",
        representation_updated=False,
    )
    action_probability = torch.sigmoid(
        apply_meter_deploy(collected["action"]["final"], action_calibration)
    )
    reason_probability = torch.sigmoid(
        apply_meter_deploy(collected["reason"]["final"], reason_calibration)
    )
    action_error = (action_probability.ge(0.5) != collected["labels_action"].bool()).sum(-1)
    reason_error = (reason_probability.ge(0.5) != collected["labels_reason"].bool()).sum(-1)
    case_order = torch.argsort(action_error + reason_error, descending=True)[:32]
    failure_cases = [
        {
            "file_name": collected["file_names"][int(index)],
            "action_error_count": int(action_error[index]),
            "reason_error_count": int(reason_error[index]),
            "action_probability": action_probability[index].tolist(),
            "reason_probability": reason_probability[index].tolist(),
            "action_target": collected["labels_action"][index].tolist(),
            "reason_target": collected["labels_reason"][index].tolist(),
        }
        for index in case_order
    ]
    mechanism = collected["mechanism"]
    evidence_cases = []
    if mechanism:
        support_map = mechanism["factor_support_map"]
        counter_map = mechanism["factor_counter_map"]
        reliability = mechanism["factor_reliability"]
        for index in range(min(32, support_map.shape[0])):
            factor_id = int(reliability[index].argmax())
            evidence_cases.append({
                "file_name": collected["file_names"][index],
                "factor_id": factor_id,
                "reliability": float(reliability[index, factor_id]),
                "support_peak_patch": int(support_map[index, factor_id].argmax()),
                "counter_peak_patch": int(counter_map[index, factor_id].argmax()),
                "support_null": float(mechanism["factor_support_null"][index, factor_id]),
                "counter_null": float(mechanism["factor_counter_null"][index, factor_id]),
            })
    cf = _test_counterfactual_diagnostic(
        model,
        dataset,
        device,
        batch_size=int(cfg["training"].get("batch_size", 6)),
        num_workers=int(cfg["data"].get("num_workers", 4)),
        max_samples=int(cfg["counterfactual"].get("diagnostic_test_samples", 128)),
        progress=progress,
        selected_mass=float(cfg["counterfactual"].get("selected_mass", 0.60)),
        max_patches=int(cfg["counterfactual"].get("max_patches", 28)),
        minimum_patches=int(cfg["counterfactual"].get("minimum_patches", 4)),
    )
    return {
        "summary": summary,
        "branches": branch_metrics(collected),
        "logits": {
            "action_final_raw_test": collected["action"]["final"],
            "reason_final_raw_test": collected["reason"]["final"],
            "action_visual_test": collected["action"]["visual"],
            "action_semantic_test": collected["action"]["semantic"],
            "action_peer_test": collected["action"]["peer"],
            "reason_calalign_test": collected["reason"]["calalign"],
            "reason_global_test": collected["reason"]["global"],
            "reason_local_test": collected["reason"]["local"],
            "reason_mix_test": collected["reason"]["mix"],
        },
        "labels": {"action_test": collected["labels_action"], "reason_test": collected["labels_reason"]},
        "file_names": collected["file_names"],
        "factor_stats": stats,
        "evidence_maps_stats": stats,
        "selector_stats": {key: stats[key] for key in ("selector_mean", "selector_std", "selector_min", "selector_max", "semantic_visual_rms_ratio", "semantic_contribution_rms", "factor_contribution_sum_error")},
        "reason_view_stats": {key: stats[key] for key in ("reason_mix_gate_mean", "reason_mix_gate_std", "reason_annotation_rms", "reason_global_local_rms")},
        "counterfactual_stats": cf,
        "failure_cases": failure_cases,
        "evidence_cases": evidence_cases,
        "factor_reliability_mean": stats.get("reliability_mean", 0.0),
        "factor_support_mean": stats.get("support_map_sum_mean", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_audit_samples", type=int, default=0)
    parser.add_argument("--max_calib_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--use_mock_dino", action="store_true")
    parser.add_argument("--require_ready", action="store_true")
    parser.add_argument("--worktree_root", default=".")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 12
    try:
        run(args)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("METER_OOM_RETRY", flush=True)
            raise SystemExit(86)
        raise


if __name__ == "__main__":
    main()
