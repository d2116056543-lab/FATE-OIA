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
    file_hash,
    save_checkpoint,
    save_epoch_artifacts,
    state_hash,
    write_json,
    load_checkpoint,
)
from fate_oia.utils.meter_config import load_meter_config
from fate_oia.utils.meter_posthoc_calibration import METERCalibrationResult, fit_train_calib_deploy_theta


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _loader(dataset: METERDataset, indices: list[int], cfg: dict[str, Any], *, shuffle: bool) -> DataLoader:
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
    return DataLoader(Subset(dataset, indices), **kwargs)


def _move_grounding(batch: dict[str, Any], device: torch.device, reason_dim: int = 21) -> dict[str, Tensor]:
    raw = batch.get("meter_grounding")
    if raw is None:
        raise RuntimeError("METER training batch is missing signed grounding targets")
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in raw.items()}


def _counterfactual_event(
    model: METEROIAModel,
    field: dict[str, Any],
    output: dict[str, Any],
    progress: float,
    *,
    action_target: Tensor | None = None,
    max_patches: int = 28,
    minimum_patches: int = 4,
) -> dict[str, Tensor | int | float]:
    """Same-field selected/control deletion; no DINO call and no feature cache."""
    maps = output["factor_support_map"].detach()
    batch, factors, patches = maps.shape
    if batch == 0:
        return {"valid_count": 0, "selected_effect": output["factor_support_score"].new_zeros(()), "control_effect": output["factor_support_score"].new_zeros(())}
    selected_count = min(max(int(minimum_patches), int(max_patches)), max(patches // 8, int(minimum_patches)))
    selected_count = min(selected_count, patches - 1)
    chosen_factor = output["factor_reliability"].detach().argmax(dim=1)
    token_mean = field["patch_tokens_by_layer"].mean(dim=1)
    token_norm = token_mean.float().square().mean(dim=-1).sqrt()
    height, width = 45, 80
    yy, xx = torch.meshgrid(torch.arange(height, device=maps.device), torch.arange(width, device=maps.device), indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)

    def select_control(selected: Tensor, row: int) -> Tensor:
        available = torch.ones(patches, dtype=torch.bool, device=maps.device)
        available[selected] = False
        selected_coords = coords[selected].float().mean(dim=0)
        distance = (coords.float() - selected_coords).abs().sum(dim=-1)
        norm_delta = (token_norm[row] - token_norm[row, selected].mean()).abs()
        score = norm_delta + 0.01 * distance
        score = score.masked_fill(~available, float("inf"))
        return torch.topk(score, k=selected_count, largest=False).indices

    def make_mask(score_maps: Tensor, factor_ids: Tensor) -> Tensor:
        mask = torch.zeros(batch, patches, dtype=torch.bool, device=maps.device)
        for row in range(batch):
            mask[row, torch.topk(score_maps[row, factor_ids[row]], k=selected_count, largest=True).indices] = True
        return mask

    support_mask = make_mask(maps, chosen_factor)
    counter_mask = make_mask(output["factor_counter_map"].detach(), chosen_factor)
    wrong_factor = (chosen_factor + 1) % factors
    wrong_mask = make_mask(maps, wrong_factor)
    support_control_mask = torch.stack([select_control(torch.where(support_mask[row])[0], row) for row in range(batch)])
    counter_control_mask = torch.stack([select_control(torch.where(counter_mask[row])[0], row) for row in range(batch)])
    support_control_mask_full = torch.zeros_like(support_mask)
    counter_control_mask_full = torch.zeros_like(counter_mask)
    support_control_mask_full.scatter_(1, support_control_mask, True)
    counter_control_mask_full.scatter_(1, counter_control_mask, True)

    def delete_field(mask: Tensor) -> dict[str, Any]:
        patch = field["patch_tokens_by_layer"]
        deleted_layers: list[Tensor] = []
        for layer in range(patch.shape[1]):
            tokens = patch[:, layer]
            image = tokens.transpose(1, 2).reshape(batch, patch.shape[-1], height, width)
            neighbor = torch.nn.functional.avg_pool2d(image, kernel_size=3, stride=1, padding=1).reshape(batch, patch.shape[-1], patches).transpose(1, 2)
            deleted_layers.append(torch.where(mask.unsqueeze(-1), neighbor, tokens))
        result = dict(field)
        result["patch_tokens_by_layer"] = torch.stack(deleted_layers, dim=1)
        return result

    selected_output = model.decode_from_field(delete_field(support_mask), progress=progress)
    control_output = model.decode_from_field(delete_field(support_control_mask_full), progress=progress)
    counter_output = model.decode_from_field(delete_field(counter_mask), progress=progress)
    counter_control_output = model.decode_from_field(delete_field(counter_control_mask_full), progress=progress)
    wrong_output = model.decode_from_field(delete_field(wrong_mask), progress=progress)
    row = torch.arange(batch, device=maps.device)
    factor = chosen_factor
    selected_effect = output["factor_support_score"][row, factor] - selected_output["factor_support_score"][row, factor]
    control_effect = output["factor_support_score"][row, factor] - control_output["factor_support_score"][row, factor]
    wrong_effect = output["factor_support_score"][row, factor] - wrong_output["factor_support_score"][row, factor]
    counter_effect = output["factor_counter_score"][row, factor] - counter_output["factor_counter_score"][row, factor]
    counter_control_effect = output["factor_counter_score"][row, factor] - counter_control_output["factor_counter_score"][row, factor]
    action_count = output["action_logits_final"].shape[1]
    if action_target is None:
        target_mask = F.one_hot(torch.arange(batch, device=maps.device) % action_count, num_classes=action_count).to(dtype=torch.bool)
    else:
        target_mask = action_target.detach().float() > 0.5
        empty = ~target_mask.any(dim=1)
        if bool(empty.any()):
            fallback = F.one_hot(action_target.detach().float().argmax(dim=1), num_classes=action_count).to(dtype=torch.bool)
            target_mask = torch.where(empty.unsqueeze(1), fallback, target_mask)
    wrong_mask = ~target_mask
    def masked_action_effect(original: Tensor, deleted: Tensor, mask: Tensor) -> Tensor:
        difference = original - deleted
        return (difference * mask.to(dtype=difference.dtype)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(dtype=difference.dtype)
    target_action_effect = masked_action_effect(output["action_logits_final"], selected_output["action_logits_final"], target_mask)
    control_action_effect = masked_action_effect(output["action_logits_final"], control_output["action_logits_final"], target_mask)
    wrong_action_effect = masked_action_effect(output["action_logits_final"], wrong_output["action_logits_final"], wrong_mask)
    event = meter_counterfactual_loss(
        selected_effect,
        control_effect,
        wrong_effect,
        selected_effect,
        counter_effect,
        target_action_effect=target_action_effect,
        wrong_action_effect=wrong_action_effect,
    )
    return {
        **event,
        "valid_count": int(batch),
        "selected_patch_count": int(selected_count),
        "counter_patch_count": int(selected_count),
        "support_control_effect": control_effect.detach(),
        "support_selected_effect": selected_effect.detach(),
        "counter_selected_effect": counter_effect.detach(),
        "counter_control_effect": counter_control_effect.detach(),
        "target_action_effect": target_action_effect.detach(),
        "control_action_effect": control_action_effect.detach(),
        "wrong_action_effect": wrong_action_effect.detach(),
        "selected_control_overlap": int((support_mask & support_control_mask_full).sum().item()),
        "counter_control_overlap": int((counter_mask & counter_control_mask_full).sum().item()),
    }


def _make_optimizer(model: METEROIAModel, cfg: dict[str, Any]) -> AdamW:
    groups: list[dict[str, Any]] = []
    def rule(name: str, lr_key: str, predicate: Any) -> tuple[str, str, Any]:
        return name, lr_key, predicate
    rules = [
        rule("foundation", "lr_foundation_core", lambda n: n.startswith("foundation.")),
        rule("factor_evidence", "lr_factor_evidence", lambda n: n.startswith("signed_factors.") and not n.startswith("signed_factors.meta_adapters.")),
        rule("semantic_action", "lr_semantic_action", lambda n: n.startswith("action_peer.") and not n.startswith("action_peer.selector.")),
        rule("action_selector", "lr_action_selector", lambda n: n.startswith("action_peer.selector.")),
        rule("reason_global_private", "lr_reason_global_private", lambda n: n.startswith("reason_decoder.") and any(f"reason_decoder.{x}" in n for x in ("private_queries", "global_query", "global_key", "global_value", "global_norm", "global_head"))),
        rule("reason_local_private", "lr_reason_local_private", lambda n: n.startswith("reason_decoder.") and any(f"reason_decoder.{x}" in n for x in ("local_proj", "local_norm", "factor_proj", "action_proj", "local_head", "mix_gate"))),
        rule("reason_annotation", "lr_reason_annotation", lambda n: n.startswith("reason_decoder.annotation_head.")),
        rule("meta_adapter", "lr_meta_adapters", lambda n: n.startswith("signed_factors.meta_adapters.")),
        rule("pu_private", "lr_pu_private", lambda n: n == "reason_decoder.tail_gain"),
    ]
    assigned: set[int] = set()
    for owner, lr_key, predicate in rules:
        parameters = [p for name, p in model.named_parameters() if predicate(name) and p.requires_grad]
        parameters = [p for p in parameters if id(p) not in assigned]
        for p in parameters:
            assigned.add(id(p))
        if parameters:
            groups.append({"params": parameters, "lr": float(cfg["training"].get(lr_key, 2e-4)), "name": owner})
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in assigned:
            assigned.add(id(p))
            groups.append({"params": [p], "lr": float(cfg["training"].get("lr_factor_evidence", 2e-4)), "name": "unclassified"})
    return AdamW(groups, weight_decay=float(cfg["training"].get("weight_decay", 0.05)))


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
) -> tuple[Tensor, dict[str, Tensor], dict[str, Any]]:
    action_target = batch["action"].to(device, non_blocking=True)
    reason_target = batch["reason"].to(device, non_blocking=True)
    confidence = output["factor_reliability"].detach()
    weights = cfg["loss_weights"]
    action = meter_action_loss(output, action_target, weights)
    reason = meter_reason_loss(output, reason_target, confidence, weights)
    grounding_target = _move_grounding(batch, device)
    grounding = meter_grounding_loss(output, grounding_target)
    private_probability = torch.sigmoid(output["reason_logits_global"].detach())
    factor_probability = output["factor_reliability"].detach()
    observability = grounding_target["factor_observability"].detach()
    pu_score = meter_pu_score(private_probability, factor_probability, 1.0 - output["factor_uncertainty"].detach(), observability)
    pu_lambda = pu_lambda.to(device=device, dtype=reason_target.dtype)
    pu = meter_private_pu_loss(output["reason_logits_final"], reason_target, pu_score, pu_lambda)
    if progress < float(cfg["training"].get("ramp_fraction", 0.10)):
        ramp = progress / max(float(cfg["training"].get("ramp_fraction", 0.10)), 1e-6)
    else:
        ramp = 1.0
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
        + (float(weights.get("grounding_start", 0.02)) + float(ramp) * (float(weights.get("grounding_end", 0.10)) - float(weights.get("grounding_start", 0.02)))) * grounding["total"]
        + float(cfg["pu"].get("max_lambda", 0.0)) * pu
        + float(weights.get("counterfactual_end", 0.03)) * float(ramp) * cf["total"]
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


def _fit_calibration(model: METEROIAModel, loader: DataLoader, device: torch.device, progress: float) -> METERCalibrationResult:
    collected = collect_outputs(model, loader, device, progress=progress)
    action = fit_train_calib_deploy_theta(collected["action"]["final"], collected["labels_action"], model_state_hash=state_hash(model))
    reason = fit_train_calib_deploy_theta(collected["reason"]["final"], collected["labels_reason"], model_state_hash=state_hash(model))
    return METERCalibrationResult(
        theta=torch.cat([action.theta, reason.theta]),
        model_state_hash_before=action.model_state_hash_before,
        model_state_hash_after=reason.model_state_hash_after,
        fit_split="train_calib",
        representation_updated=False,
    )


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    cfg = load_meter_config(config_path)
    if args.batch_size:
        cfg["training"]["batch_size"] = int(args.batch_size)
    grad_accum = int(args.gradient_accumulation_steps or cfg["training"].get("gradient_accumulation_steps", 1))
    if grad_accum < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.require_ready:
        ready = Path(args.worktree_root or ".") / ".review" / "METER_OIA_V1_PRE_PILOT_READY.json"
        if not ready.exists():
            raise RuntimeError("METER pre-pilot readiness artifact is required before training")
    device = torch.device(args.device)
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
    train_loader = _loader(dataset, main_indices, cfg, shuffle=True)
    audit_loader = _loader(dataset, audit_indices, cfg, shuffle=False)
    calib_loader = _loader(dataset, calib_indices, cfg, shuffle=False)
    meta = METERMetaUtility(
        factors=cfg["model"]["reason_dim"], virtual_lr=cfg["meta"]["virtual_lr"],
        ema_old_weight=cfg["meta"]["ema_old_weight"], ema_new_weight=cfg["meta"]["ema_new_weight"],
        lower=cfg["meta"]["utility_lower"], upper=cfg["meta"]["utility_upper"],
    )
    source_hash = str(cfg["audit"]["require_source_sha"])
    config_hash = file_hash(config_path)
    schema_hash = _sha(Path("configs/meter_factor_schema.yaml").read_text(encoding="utf-8") + Path("configs/meter_grounding_schema.yaml").read_text(encoding="utf-8"))
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head, git_branch = "unknown", "unknown"
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(output_dir / "config_resolved.yaml.json", cfg)
    write_json(output_dir / "source_fingerprint.json", {"git_head": git_head, "branch": git_branch, "base_sha": source_hash, "config_hash": config_hash, "schema_hash": schema_hash})
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
        "experiment": cfg["experiment"], "command_line": " ".join(os.sys.argv), "git_head": git_head, "branch": git_branch, "github_main_baseline_head": source_hash, "source_hash": source_hash,
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
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, expected_config_hash=config_hash, expected_source_hash=source_hash, expected_schema_hash=schema_hash)
        global_step = int(payload.get("optimizer_step", 0))
        start_epoch = int(payload.get("epoch", -1)) + 1
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
            calibration = METERCalibrationResult(theta=torch.as_tensor(calibration_payload["theta"]), model_state_hash_before="", model_state_hash_after="", fit_split="train_calib", representation_updated=False)
        best_joint = float(payload.get("meta_state", {}).get("best_joint", float("-inf")))
        best_branch_metrics = {str(key): float(value) for key, value in payload.get("meta_state", {}).get("best_branch_metrics", {}).items()}
    with torch.no_grad():
        model.meta_share_weight.copy_(meta.omega.to(device=model.meta_share_weight.device, dtype=model.meta_share_weight.dtype))
    iterator = iter(train_loader)
    amp_enabled = bool(cfg["training"].get("bf16", False)) and device.type == "cuda"
    amp_context = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled) if device.type in {"cuda", "cpu"} else nullcontext()
    meta_every_updates = int(cfg["meta"].get("pilot_every_updates", 40) if int(args.epochs) <= 3 else cfg["meta"].get("full_every_updates", 100))
    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        model.foundation.dino.eval()
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        for micro in range(micro_batches_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            images = batch["image"].to(device, non_blocking=True)
            is_update = ((micro + 1) % grad_accum == 0) or micro == micro_batches_per_epoch - 1
            with amp_context():
                field = model.encode_images(images)
                progress = min(global_step / max(total_updates, 1), 1.0)
                output = model.decode_from_field(field, progress=progress)
                counterfactual = None
                if is_update and (global_step + 1) % int(cfg["counterfactual"].get("every_optimizer_updates", 8)) == 0:
                    counterfactual = _counterfactual_event(
                        model,
                        field,
                        output,
                        progress,
                        action_target=batch["action"].to(device),
                        max_patches=int(cfg["counterfactual"].get("max_patches", 28)),
                        minimum_patches=int(cfg["counterfactual"].get("minimum_patches", 4)),
                    )
                    append_jsonl(output_dir / "counterfactual_events.jsonl", {"epoch": epoch, "step": global_step, **{key: (float(value.detach().cpu()) if isinstance(value, Tensor) and value.ndim == 0 else (value.detach().cpu().tolist() if isinstance(value, Tensor) else value)) for key, value in counterfactual.items() if key != "total"}})
                total, parts, diagnostics = _losses(model, output, batch, cfg, progress, device, pu_lambda=pu_lambda, counterfactual=counterfactual)
            (total / grad_accum).backward()
            if not is_update:
                continue
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 1.0))).detach().cpu())
            owner_grad_norms: dict[str, float] = {}
            owner_delta_estimates: dict[str, float] = {}
            for group in optimizer.param_groups:
                owner = str(group.get("name", "unclassified"))
                squared = sum((parameter.grad.detach().float().square().sum() for parameter in group["params"] if parameter.grad is not None), torch.zeros((), device=device))
                owner_grad_norms[owner] = float(squared.sqrt().cpu())
                owner_delta_estimates[owner] = float((squared.sqrt() * float(group["lr"])).cpu())
            optimizer.step()
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
                    return meter_reason_loss(virtual, batch["reason"].to(device), virtual["factor_reliability"], cfg["loss_weights"])["total"]
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
                    "relative_utility": float(event.relative_utility),
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
                "loss_breakdown": {
                    "action": {key: float(value.detach().cpu()) for key, value in diagnostics["action"].items() if isinstance(value, Tensor)},
                    "reason": {key: float(value.detach().cpu()) for key, value in diagnostics["reason"].items() if isinstance(value, Tensor)},
                    "grounding": {key: float(value.detach().cpu()) for key, value in diagnostics["grounding"].items() if isinstance(value, Tensor)},
                    "counterfactual": {key: float(value.detach().cpu()) for key, value in diagnostics["counterfactual"].items() if isinstance(value, Tensor) and value.ndim == 0},
                },
            })
            if global_step % int(cfg["runtime"].get("print_every_optimizer_updates", 50)) == 0:
                print(json.dumps({"meter_batch": {"epoch": epoch, "step": global_step, "loss": float(total.detach().cpu()), "action": float(parts["action"].detach().cpu()), "reason": float(parts["reason"].detach().cpu()), "grounding": float(parts["grounding"].detach().cpu()), "pu": float(parts["pu"].detach().cpu()), "grad_norm": grad_norm}}, sort_keys=True), flush=True)
        audit_outputs = collect_outputs(model, audit_loader, device, progress=1.0)
        audit_mechanism = audit_outputs.get("mechanism", {})
        pu_audit_state = meter_hidden_positive_audit(
            torch.sigmoid(audit_outputs["reason"]["global"]),
            audit_mechanism.get("factor_reliability", torch.zeros_like(audit_outputs["reason"]["global"])),
            audit_outputs["labels_reason"],
            hidden_fraction=float(cfg["pu"].get("hidden_positive_fraction", 0.30)),
            min_positive_count=int(cfg["pu"].get("min_positive_count", 20)),
            seed=int(cfg["splits"].get("seed", 20260728)) + epoch,
        )
        if epoch >= 0:
            pu_lambda = torch.as_tensor(pu_audit_state["lambda"], device=device, dtype=torch.float32).clamp_min(0.0).clamp_max(float(cfg["pu"].get("max_lambda", 0.15)))
        append_jsonl(output_dir / "pu_audit.jsonl", {"epoch": epoch, **pu_audit_state, "active_lambda": pu_lambda.detach().cpu().tolist()})
        calibration = _fit_calibration(model, calib_loader, device, progress=1.0)
        test_result = evaluate_test(model, cfg, device, args, progress=1.0, calibration=calibration)
        raw = test_result["summary"]["metrics_raw"]
        deploy = test_result["summary"]["metrics_deploy"]
        joint = float(deploy.get("deploy_joint", 0.0))
        append_jsonl(output_dir / "metrics_summary.jsonl", {"epoch": epoch, "metrics_raw": raw, "metrics_deploy": deploy, "runtime_sec": time.time() - epoch_start, "dino_calls": model.foundation.ordinary_dino_calls})
        append_jsonl(output_dir / "mechanism_stats.jsonl", {"epoch": epoch, "branch_metrics": test_result["branches"], "factor_stats": test_result["factor_stats"], "selector_stats": test_result["selector_stats"], "reason_view_stats": test_result["reason_view_stats"], "factor_reliability_mean": float(test_result["factor_reliability_mean"]), "factor_support_mean": float(test_result["factor_support_mean"]), "pu_active_labels": pu_audit_state["active_labels"]})
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
            "failure_cases.jsonl": {"epoch": epoch, "available": True, "note": "case mining is derived from test branch logits; raw tensors are saved for reproducible case selection"},
            "evidence_cases.jsonl": {"epoch": epoch, "available": True, "file_count": len(test_result["file_names"]), "factor_stats": test_result["factor_stats"]},
            "calibration.json": {"theta": calibration.theta.tolist(), "fit_split": calibration.fit_split, "state_hash_before": calibration.model_state_hash_before, "state_hash_after": calibration.model_state_hash_after},
        }
        save_epoch_artifacts(output_dir, epoch, metrics_raw=raw, metrics_deploy=deploy, branch_metrics=test_result["branches"], logits=test_result["logits"], labels=test_result["labels"], diagnostics=diagnostics, file_names=test_result["file_names"])
        checkpoint_runtime = {"batch_size": cfg["training"].get("batch_size"), "gradient_accumulation_steps": grad_accum, "effective_batch": int(cfg["training"].get("batch_size", 6)) * grad_accum, "runtime_profile": runtime_profile}
        checkpoint_meta = {"omega": meta.omega, "utility_ema": meta.utility_ema, "cursor": meta.cursor, "best_joint": best_joint, "best_branch_metrics": best_branch_metrics}
        save_checkpoint(output_dir / "checkpoint_latest.pth", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, micro_step=0, optimizer_step=global_step, runtime_profile=checkpoint_runtime, meta_state=checkpoint_meta, pu_state={"lambda": pu_lambda.detach().cpu(), "audit": pu_audit_state}, calibration={"theta": calibration.theta.detach().cpu()}, config_hash=config_hash, source_hash=source_hash, schema_hash=schema_hash)
        if joint > best_joint:
            best_joint = joint
            checkpoint_meta["best_joint"] = best_joint
            save_checkpoint(output_dir / "checkpoint_best_test_deploy_joint.pth", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, micro_step=0, optimizer_step=global_step, runtime_profile=checkpoint_runtime, meta_state=checkpoint_meta, pu_state={"lambda": pu_lambda.detach().cpu(), "audit": pu_audit_state}, calibration={"theta": calibration.theta.detach().cpu()}, config_hash=config_hash, source_hash=source_hash, schema_hash=schema_hash)
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
                save_checkpoint(output_dir / best_checkpoint_names[metric_name], model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, micro_step=0, optimizer_step=global_step, runtime_profile=checkpoint_runtime, meta_state=checkpoint_meta, pu_state={"lambda": pu_lambda.detach().cpu(), "audit": pu_audit_state}, calibration={"theta": calibration.theta.detach().cpu()}, config_hash=config_hash, source_hash=source_hash, schema_hash=schema_hash)
        print(json.dumps({"epoch": epoch, "test": {"Act_mF1": deploy.get("Act_mF1"), "Act_oF1": deploy.get("Act_oF1"), "Exp_mF1": deploy.get("Exp_mF1"), "Exp_oF1": deploy.get("Exp_oF1"), "deploy_joint": joint}, "best_joint": best_joint}, sort_keys=True), flush=True)


@torch.no_grad()
def _test_counterfactual_diagnostic(model: METEROIAModel, dataset: METERDataset, device: torch.device, *, batch_size: int, num_workers: int, max_samples: int, progress: float, max_patches: int, minimum_patches: int) -> dict[str, Any]:
    loader = DataLoader(Subset(dataset, list(range(min(len(dataset), max_samples)))), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=2)
    records: list[dict[str, Any]] = []
    seen = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        field = model.encode_images(images)
        output = model.decode_from_field(field, progress=progress)
        event = _counterfactual_event(model, field, output, progress, action_target=batch["action"].to(device), max_patches=max_patches, minimum_patches=minimum_patches)
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
    cf = _test_counterfactual_diagnostic(
        model,
        dataset,
        device,
        batch_size=int(cfg["training"].get("batch_size", 6)),
        num_workers=int(cfg["data"].get("num_workers", 4)),
        max_samples=int(cfg["counterfactual"].get("diagnostic_test_samples", 128)),
        progress=progress,
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
