from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from fate_oia.datasets.aie_splits import stable_split_ids, write_split_manifest
from fate_oia.engine.train_aie_oia import (
    build_model,
    compatible_checkpoint_state_dict,
    load_config,
    make_dataset,
    make_loader,
)
from fate_oia.models.aie_trainable_decision_model import AIETrainableDecisionModel
from fate_oia.utils.aie_artifacts import write_json
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.losses.vetra_strong_rank_losses import action_smooth_ap_loss


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def soft_macro_f1_loss(
    logits: torch.Tensor, target: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    probability = torch.sigmoid(logits / float(temperature))
    true_positive = (probability * target).sum(0)
    denominator = probability.sum(0) + target.sum(0)
    return 1.0 - ((2.0 * true_positive + 1e-6) / (denominator + 1e-6)).mean()


def global_soft_f1_stats(
    logits: torch.Tensor, target: torch.Tensor, temperature: float = 1.0
) -> dict[str, torch.Tensor]:
    probability = torch.sigmoid(logits.float() / float(temperature))
    true_positive = (probability * target.float()).sum(0)
    return {
        "numerator": 2.0 * true_positive + 1e-6,
        "denominator": probability.sum(0) + target.float().sum(0) + 1e-6,
        "temperature": torch.as_tensor(float(temperature), device=logits.device),
    }


def global_soft_f1_linearized_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Exact full-set soft-F1 gradient expressed as a batchwise linear form."""
    temperature = float(stats.get("temperature", torch.tensor(1.0)).detach().cpu())
    probability = torch.sigmoid(logits.float() / temperature)
    numerator = stats["numerator"].to(probability)
    denominator = stats["denominator"].to(probability)
    coefficient = -(
        2.0 * target.float() * denominator - numerator
    ) / denominator.square().clamp_min(1e-12)
    coefficient = coefficient / probability.shape[1]
    return (coefficient.detach() * probability).sum()


def positive_boundary_hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_counts: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    counts = positive_counts.to(logits).clamp_min(1.0)
    per_label = (torch.relu(float(margin) - logits.float()) * target.float()).sum(0) / counts
    available = positive_counts.to(logits) > 0
    if not bool(available.any()):
        return logits.sum() * 0.0
    return per_label[available].mean()


def contradiction_weighted_reason_calibration_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    contradiction: torch.Tensor,
    *,
    reliable_negative_min: float = 0.6,
) -> torch.Tensor:
    """Proper logistic loss on observed positives and grounded negatives only.

    An unobserved reason is not a negative in BDD-OIA. It contributes only when
    the frozen visual/grammar path supplies sufficiently strong contradictory
    evidence. Contradiction weights are detached so this decision-only stage
    cannot teach the evidence path to manufacture easy negatives.
    """
    logits = logits.float()
    target = target.float()
    confidence = contradiction.detach().float().clamp(0.0, 1.0)
    negative_weight = (
        (target < 0.5)
        & (confidence >= float(reliable_negative_min))
    ).to(logits) * confidence

    positive_count = target.sum(0)
    negative_count = negative_weight.sum(0)
    positive = (F.softplus(-logits) * target).sum(0) / positive_count.clamp_min(1.0)
    negative = (F.softplus(logits) * negative_weight).sum(0) / negative_count.clamp_min(1.0)
    positive_available = positive_count > 0
    negative_available = negative_count > 0
    terms = []
    if bool(positive_available.any()):
        terms.append(positive[positive_available].mean())
    if bool(negative_available.any()):
        terms.append(negative[negative_available].mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def reason_label_pair_ranking_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    contradiction: torch.Tensor,
    *,
    reliable_negative_min: float = 0.6,
    margin: float = 0.2,
) -> torch.Tensor:
    """Rank each observed positive above grounded contradictory reasons."""
    logits = logits.float()
    target = target.float()
    confidence = contradiction.detach().float().clamp(0.0, 1.0)
    positive = target > 0.5
    negative_weight = (
        (~positive) & (confidence >= float(reliable_negative_min))
    ).to(logits) * confidence
    pair_weight = positive[:, :, None].to(logits) * negative_weight[:, None, :]
    pair_loss = F.softplus(
        float(margin) - (logits[:, :, None] - logits[:, None, :])
    ) * pair_weight
    per_sample_count = pair_weight.sum((1, 2))
    available = per_sample_count > 0
    if not bool(available.any()):
        return logits.sum() * 0.0
    return (pair_loss.sum((1, 2))[available] / per_sample_count[available]).mean()


def reason_tail_mask(positive_counts: torch.Tensor, max_positive: int) -> torch.Tensor:
    return (positive_counts > 0) & (positive_counts <= int(max_positive))


def decision_loss(output: dict[str, Any], action: torch.Tensor, reason: torch.Tensor) -> dict[str, torch.Tensor]:
    action_logits = output["action_logits_decision"]
    reason_logits = output["reason_logits_decision"]
    losses = {
        "action_bce": F.binary_cross_entropy_with_logits(action_logits, action),
        "action_soft_f1": soft_macro_f1_loss(action_logits, action),
        "reason_bce": F.binary_cross_entropy_with_logits(reason_logits, reason),
        "reason_soft_f1": soft_macro_f1_loss(reason_logits, reason),
        "action_smooth_ap": action_smooth_ap_loss(action_logits, action, temperature=0.20),
    }
    delta = output["action_delta"].float()
    primary = output["action_logits_primary"].float()
    ratio = delta.square().mean(0).sqrt() / primary.square().mean(0).sqrt().clamp_min(1e-6)
    losses["action_delta_guard"] = torch.relu(ratio - 0.25).square().mean()
    losses["total"] = (
        losses["action_soft_f1"]
        + losses["reason_soft_f1"]
        + 0.20 * losses["action_smooth_ap"]
        + 2.0 * losses["action_delta_guard"]
    )
    return losses


@torch.no_grad()
def evaluate(model: AIETrainableDecisionModel, loader, device: torch.device) -> dict[str, Any]:
    storage = {key: [] for key in ("action", "reason", "action_target", "reason_target")}
    delta_square = torch.zeros(4, dtype=torch.float64)
    primary_square = torch.zeros(4, dtype=torch.float64)
    named_sum = 0.0
    sample_count = 0
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images)
        storage["action"].append(output["action_logits_decision"].float().cpu())
        storage["reason"].append(output["reason_logits_decision"].float().cpu())
        storage["action_target"].append(batch["action"].float().cpu())
        storage["reason_target"].append(batch["reason"].float().cpu())
        delta_square += output["action_delta"].double().square().sum(0).cpu()
        primary_square += output["action_logits_primary"].double().square().sum(0).cpu()
        batch_size = images.shape[0]
        named_sum += float(output.get("named_coverage", torch.tensor(0.0)).detach().cpu()) * batch_size
        sample_count += batch_size
    joined = {key: torch.cat(values) for key, values in storage.items()}
    metrics = aie_branch_metrics(
        joined["action"], joined["reason"], joined["action_target"], joined["reason_target"]
    )
    delta_ratio = torch.sqrt(delta_square / max(sample_count, 1)) / torch.sqrt(
        primary_square / max(sample_count, 1)
    ).clamp_min(1e-12)
    return {
        **metrics,
        "sample_count": sample_count,
        "named_coverage": named_sum / max(sample_count, 1),
        "action_delta_to_primary_rms_by_action": delta_ratio.tolist(),
        "action_delta_to_primary_rms_mean": float(delta_ratio.mean()),
        "action_scales": model.action_scales.detach().cpu().tolist(),
        "reason_scales": model.reason_scales.detach().cpu().tolist(),
        "threshold_prob": model.threshold_prob.detach().cpu().tolist(),
    }


def decision_state(model: AIETrainableDecisionModel) -> dict[str, torch.Tensor]:
    return {
        "action_scale_raw": model.action_scale_raw.detach().cpu(),
        "reason_scale_raw": model.reason_scale_raw.detach().cpu(),
        "threshold_raw": model.threshold_raw.detach().cpu(),
    }


def load_decision_state(model: AIETrainableDecisionModel, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        model.action_scale_raw.copy_(state["action_scale_raw"].to(model.action_scale_raw))
        if "reason_scale_raw" in state:
            model.reason_scale_raw.copy_(state["reason_scale_raw"].to(model.reason_scale_raw))
        model.threshold_raw.copy_(state["threshold_raw"].to(model.threshold_raw))


@torch.no_grad()
def collect_global_stats(
    model: AIETrainableDecisionModel, loader, device: torch.device, temperature: float
):
    totals = {
        "action_probability": torch.zeros(4, device=device),
        "action_true_positive": torch.zeros(4, device=device),
        "action_target": torch.zeros(4, device=device),
        "reason_probability": torch.zeros(21, device=device),
        "reason_true_positive": torch.zeros(21, device=device),
        "reason_target": torch.zeros(21, device=device),
    }
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True).float()
        reason = batch["reason"].to(device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images)
        action_probability = torch.sigmoid(output["action_logits_decision"].float() / temperature)
        reason_probability = torch.sigmoid(output["reason_logits_decision"].float() / temperature)
        totals["action_probability"] += action_probability.sum(0)
        totals["action_true_positive"] += (action_probability * action).sum(0)
        totals["action_target"] += action.sum(0)
        totals["reason_probability"] += reason_probability.sum(0)
        totals["reason_true_positive"] += (reason_probability * reason).sum(0)
        totals["reason_target"] += reason.sum(0)
    action_stats = {
        "numerator": 2.0 * totals["action_true_positive"] + 1e-6,
        "denominator": totals["action_probability"] + totals["action_target"] + 1e-6,
        "temperature": torch.as_tensor(temperature, device=device),
        "target_count": totals["action_target"],
    }
    reason_stats = {
        "numerator": 2.0 * totals["reason_true_positive"] + 1e-6,
        "denominator": totals["reason_probability"] + totals["reason_target"] + 1e-6,
        "temperature": torch.as_tensor(temperature, device=device),
        "target_count": totals["reason_target"],
    }
    return action_stats, reason_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--threshold-lr", type=float, default=10.0)
    parser.add_argument("--tail-positive-hinge-weight", type=float, default=0.10)
    parser.add_argument("--init-decision-checkpoint")
    parser.add_argument("--freeze-action-decision", action="store_true")
    parser.add_argument("--freeze-thresholds", action="store_true")
    parser.add_argument("--tail-reason-max-positive", type=int, default=0)
    parser.add_argument("--cv-fold-index", type=int)
    parser.add_argument("--cv-fold-count", type=int, default=5)
    parser.add_argument("--fit-all-decision-samples", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--reset-reason-thresholds", action="store_true")
    parser.add_argument("--reason-scale", type=float, default=0.6)
    parser.add_argument("--train-reason-scales", action="store_true")
    parser.add_argument("--reason-scale-lr", type=float, default=0.05)
    parser.add_argument("--pu-calibration-weight", type=float, default=0.0)
    parser.add_argument("--reason-pair-weight", type=float, default=0.0)
    parser.add_argument("--reliable-negative-min", type=float, default=0.6)
    parser.add_argument("--reason-pair-margin", type=float, default=0.2)
    parser.add_argument("--reason-action-scale", type=float, default=0.0)
    parser.add_argument("--f1-temperature-start", type=float, default=0.5)
    parser.add_argument("--f1-temperature-min", type=float, default=0.15)
    parser.add_argument("--f1-temperature-decay", type=float, default=0.85)
    parser.add_argument("--max-calib-samples", type=int)
    parser.add_argument("--max-audit-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["data"]["split_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("cpu_threads", 16)))
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    source_path = Path(args.source_checkpoint)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    base_model = build_model(cfg, device)
    base_model.load_state_dict(compatible_checkpoint_state_dict(base_model, source["model"]), strict=True)
    model = AIETrainableDecisionModel(
        base_model,
        reason_scale=args.reason_scale,
        reason_action_scale=args.reason_action_scale,
    ).to(device)
    if args.init_decision_checkpoint:
        initial_decision = torch.load(
            args.init_decision_checkpoint, map_location="cpu", weights_only=False
        )
        load_decision_state(model, initial_decision["decision_state"])
    if args.reset_reason_thresholds:
        with torch.no_grad():
            initial = torch.full((21,), 0.5, device=device)
            normalized = (
                (initial - model.threshold_lower[4:])
                / (model.threshold_upper[4:] - model.threshold_lower[4:])
            ).clamp(1e-6, 1.0 - 1e-6)
            model.threshold_raw[4:].copy_(torch.logit(normalized))
    frozen_action_scale = model.action_scale_raw.detach().clone()
    frozen_action_threshold = model.threshold_raw[:4].detach().clone()
    threshold_mask = None
    scale_parameter_groups = []
    if args.freeze_action_decision:
        model.action_scale_raw.requires_grad_(False)
    else:
        scale_parameter_groups.append({"params": [model.action_scale_raw], "lr": args.lr})
    if args.train_reason_scales:
        scale_parameter_groups.append({"params": [model.reason_scale_raw], "lr": args.reason_scale_lr})
    else:
        model.reason_scale_raw.requires_grad_(False)
    scale_optimizer = torch.optim.Adam(scale_parameter_groups) if scale_parameter_groups else None
    if args.freeze_thresholds:
        model.threshold_raw.requires_grad_(False)
    elif args.freeze_action_decision or args.tail_reason_max_positive > 0:
        threshold_mask = torch.cat(
            (
                torch.zeros(4, device=device)
                if args.freeze_action_decision
                else torch.ones(4, device=device),
                torch.ones(21, device=device),
            )
        )
        model.threshold_raw.register_hook(lambda gradient: gradient * threshold_mask)
    threshold_optimizer = None
    if not args.freeze_thresholds:
        threshold_optimizer = torch.optim.SGD(
            (model.threshold_raw,), lr=args.threshold_lr, momentum=0.9
        )

    train_dataset = make_dataset(cfg, "train")
    train_names = [sample.file_name for sample in train_dataset.samples]
    split = stable_split_ids(
        train_names,
        seed,
        float(cfg["data"]["train_calib_fraction"]),
        int(cfg["data"]["train_audit_count"]),
    )
    index = {sample.file_name: position for position, sample in enumerate(train_dataset.samples)}
    if args.fit_all_decision_samples and args.cv_fold_index is not None:
        raise ValueError("fit-all-decision-samples and cv-fold-index are mutually exclusive")
    if args.fit_all_decision_samples:
        calib_names = sorted([*split["train_calib"], *split["train_audit"]])
        audit_names = []
    elif args.cv_fold_index is not None:
        if not 0 <= args.cv_fold_index < args.cv_fold_count:
            raise ValueError("cv-fold-index must be in [0, cv-fold-count)")
        pool = sorted([*split["train_calib"], *split["train_audit"]])
        audit_names = [name for position, name in enumerate(pool) if position % args.cv_fold_count == args.cv_fold_index]
        calib_names = [name for position, name in enumerate(pool) if position % args.cv_fold_count != args.cv_fold_index]
    else:
        calib_names = split["train_calib"]
        audit_names = split["train_audit"]
    calib_names = calib_names[: args.max_calib_samples or None]
    audit_names = audit_names[: args.max_audit_samples or None]
    if set(calib_names) & set(audit_names):
        raise RuntimeError("train_calib and train_audit overlap")
    calib_loader = make_loader(
        Subset(train_dataset, [index[name] for name in calib_names]),
        args.batch_size,
        True,
        args.num_workers,
        cfg,
    )
    audit_loader = None
    if audit_names:
        audit_loader = make_loader(
            Subset(train_dataset, [index[name] for name in audit_names]),
            args.batch_size,
            False,
            args.num_workers,
            cfg,
        )
    write_split_manifest(
        output_dir / "split_manifest.json",
        train_names,
        seed,
        float(cfg["data"]["train_calib_fraction"]),
        int(cfg["data"]["train_audit_count"]),
    )
    run_manifest = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command_line": [sys.executable, *sys.argv],
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": file_sha256(source_path),
        "direct_image": True,
        "feature_cache_enabled": False,
        "token_compression": "none",
        "base_model_frozen": True,
        "trainable_parameters": [
            name
            for name, enabled in (
                ("action_scale_raw", not args.freeze_action_decision),
                ("reason_scale_raw", args.train_reason_scales),
                ("threshold_raw", True),
            )
            if enabled
        ],
        "train_calib_count": len(calib_names),
        "train_audit_count": len(audit_names),
        "test_read_during_training": False,
        "checkpoint_selection_split": "fixed_cv_epoch" if args.fit_all_decision_samples else "train_audit",
        "final_test_evaluations": 1,
        "reason_scale": args.reason_scale,
        "train_reason_scales": args.train_reason_scales,
        "reason_scale_lr": args.reason_scale_lr,
        "pu_calibration_weight": args.pu_calibration_weight,
        "reason_pair_weight": args.reason_pair_weight,
        "reliable_negative_min": args.reliable_negative_min,
        "reason_pair_margin": args.reason_pair_margin,
        "reason_action_scale": args.reason_action_scale,
        "init_decision_checkpoint": args.init_decision_checkpoint,
        "tail_positive_hinge_weight": args.tail_positive_hinge_weight,
        "freeze_action_decision": args.freeze_action_decision,
        "freeze_thresholds": args.freeze_thresholds,
        "tail_reason_max_positive": args.tail_reason_max_positive,
        "cv_fold_index": args.cv_fold_index,
        "cv_fold_count": args.cv_fold_count,
        "fit_all_decision_samples": args.fit_all_decision_samples,
        "skip_final_test": args.skip_final_test,
        "reset_reason_thresholds": args.reset_reason_thresholds,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    best_score = -float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        temperature = max(
            args.f1_temperature_min,
            args.f1_temperature_start * (args.f1_temperature_decay ** epoch),
        )
        action_stats, reason_stats = collect_global_stats(model, calib_loader, device, temperature)
        if threshold_mask is not None and args.tail_reason_max_positive > 0:
            threshold_mask[4:] = reason_tail_mask(
                reason_stats["target_count"], args.tail_reason_max_positive
            ).to(threshold_mask)
        model.train()
        if scale_optimizer is not None:
            scale_optimizer.zero_grad(set_to_none=True)
        if threshold_optimizer is not None:
            threshold_optimizer.zero_grad(set_to_none=True)
        running_ap = 0.0
        running_guard = 0.0
        running_reason_hinge = 0.0
        running_pu_calibration = 0.0
        running_reason_pair = 0.0
        for step, batch in enumerate(calib_loader):
            images = batch["image"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True).float()
            reason = batch["reason"].to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(images)
                action_global = global_soft_f1_linearized_loss(
                    output["action_logits_decision"], action, action_stats
                )
                reason_global = global_soft_f1_linearized_loss(
                    output["reason_logits_decision"], reason, reason_stats
                )
                smooth_ap = action_smooth_ap_loss(
                    output["action_logits_decision"], action, temperature=0.20
                ) / len(calib_loader)
                delta = output["action_delta"].float()
                primary = output["action_logits_primary"].float()
                ratio = delta.square().mean(0).sqrt() / primary.square().mean(0).sqrt().clamp_min(1e-6)
                guard = torch.relu(ratio - 0.25).square().mean() / len(calib_loader)
                reason_hinge = positive_boundary_hinge_loss(
                    output["reason_logits_decision"],
                    reason,
                    reason_stats["target_count"],
                    margin=0.2,
                )
                contradiction = output["contradiction_score"]
                pu_calibration = contradiction_weighted_reason_calibration_loss(
                    output["reason_logits_decision"],
                    reason,
                    contradiction,
                    reliable_negative_min=args.reliable_negative_min,
                ) / len(calib_loader)
                reason_pair = reason_label_pair_ranking_loss(
                    output["reason_logits_decision"],
                    reason,
                    contradiction,
                    reliable_negative_min=args.reliable_negative_min,
                    margin=args.reason_pair_margin,
                ) / len(calib_loader)
                total = (
                    action_global
                    + reason_global
                    + 0.20 * smooth_ap
                    + 2.0 * guard
                    + args.tail_positive_hinge_weight * reason_hinge
                    + args.pu_calibration_weight * pu_calibration
                    + args.reason_pair_weight * reason_pair
                )
            total.backward()
            running_ap += float(smooth_ap.detach())
            running_guard += float(guard.detach())
            running_reason_hinge += float(reason_hinge.detach())
            running_pu_calibration += float(pu_calibration.detach())
            running_reason_pair += float(reason_pair.detach())
            if step == 0 or (step + 1) % 25 == 0:
                row = {
                    "event": "trainable_decision_accumulation",
                    "epoch": epoch,
                    "step": step + 1,
                    "total_steps": len(calib_loader),
                    "action_scales": model.action_scales.detach().cpu().tolist(),
                    "reason_scales": model.reason_scales.detach().cpu().tolist(),
                    "threshold_action_mean": float(model.threshold_prob[:4].detach().mean()),
                    "threshold_reason_mean": float(model.threshold_prob[4:].detach().mean()),
                    "global_action_soft_f1": float(1.0 - (action_stats["numerator"] / action_stats["denominator"]).mean()),
                    "global_reason_soft_f1": float(1.0 - (reason_stats["numerator"] / reason_stats["denominator"]).mean()),
                }
                print(json.dumps(row), flush=True)
                append_jsonl(output_dir / "loss_components.jsonl", row)

        scale_grad = (
            float(model.action_scale_raw.grad.detach().norm())
            if model.action_scale_raw.grad is not None
            else 0.0
        )
        reason_scale_grad = (
            float(model.reason_scale_raw.grad.detach().norm())
            if model.reason_scale_raw.grad is not None
            else 0.0
        )
        threshold_grad = (
            float(model.threshold_raw.grad.detach().norm())
            if model.threshold_raw.grad is not None
            else 0.0
        )
        parameters_to_clip = []
        if not args.freeze_thresholds:
            parameters_to_clip.append(model.threshold_raw)
        if not args.freeze_action_decision:
            parameters_to_clip.append(model.action_scale_raw)
        if args.train_reason_scales:
            parameters_to_clip.append(model.reason_scale_raw)
        grad_norm = (
            float(torch.nn.utils.clip_grad_norm_(parameters_to_clip, 5.0))
            if parameters_to_clip
            else 0.0
        )
        if scale_optimizer is not None:
            scale_optimizer.step()
        if threshold_optimizer is not None:
            threshold_optimizer.step()
        if args.freeze_action_decision:
            torch.testing.assert_close(model.action_scale_raw.detach(), frozen_action_scale, atol=0.0, rtol=0.0)
            torch.testing.assert_close(model.threshold_raw[:4].detach(), frozen_action_threshold, atol=0.0, rtol=0.0)
        update_row = {
            "event": "trainable_decision_update",
            "epoch": epoch,
            "f1_temperature": temperature,
            "scale_grad_norm": scale_grad,
            "reason_scale_grad_norm": reason_scale_grad,
            "threshold_grad_norm": threshold_grad,
            "grad_norm": grad_norm,
            "global_action_soft_f1": float(1.0 - (action_stats["numerator"] / action_stats["denominator"]).mean()),
            "global_reason_soft_f1": float(1.0 - (reason_stats["numerator"] / reason_stats["denominator"]).mean()),
            "action_smooth_ap": running_ap,
            "action_delta_guard": running_guard,
            "reason_positive_boundary_hinge": running_reason_hinge,
            "reason_pu_calibration": running_pu_calibration,
            "reason_label_pair_ranking": running_reason_pair,
            "action_scales": model.action_scales.detach().cpu().tolist(),
            "reason_scales": model.reason_scales.detach().cpu().tolist(),
            "threshold_prob": model.threshold_prob.detach().cpu().tolist(),
            "threshold_prob_std": float(model.threshold_prob.detach().std()),
        }
        print(json.dumps(update_row), flush=True)
        append_jsonl(output_dir / "loss_components.jsonl", update_row)

        if audit_loader is None:
            audit = None
            score = float(epoch)
            epoch_row = {
                "event": "trainable_decision_fixed_epoch",
                "epoch": epoch,
                "selection_source": "five_fold_train_only_curve",
                "test_read_during_training": False,
            }
        else:
            audit = evaluate(model, audit_loader, device)
            score = 0.5 * float(audit["Act_mF1"]) + 0.5 * float(audit["Exp_mF1"])
            epoch_row = {"event": "trainable_decision_epoch", "epoch": epoch, "audit_joint": score, **audit}
        print(json.dumps(epoch_row), flush=True)
        append_jsonl(output_dir / "audit_metrics.jsonl", epoch_row)
        checkpoint = {
            "epoch": epoch,
            "decision_state": decision_state(model),
            "scale_optimizer": scale_optimizer.state_dict() if scale_optimizer is not None else None,
            "threshold_optimizer": threshold_optimizer.state_dict() if threshold_optimizer is not None else None,
            "audit_metrics": audit,
            "audit_joint": score if audit is not None else None,
            "source_checkpoint": str(source_path.resolve()),
            "source_checkpoint_sha256": run_manifest["source_checkpoint_sha256"],
            "reason_scale": args.reason_scale,
            "reason_action_scale": args.reason_action_scale,
        }
        torch.save(checkpoint, output_dir / "checkpoint_latest.pth")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(checkpoint, output_dir / "checkpoint_best_train_audit.pth")

    best = torch.load(output_dir / "checkpoint_best_train_audit.pth", map_location="cpu", weights_only=False)
    if args.skip_final_test:
        result = {
            "best_train_audit_epoch": best_epoch,
            "best_train_audit_joint": best_score,
            "test_evaluation_count": 0,
        }
        write_json(output_dir / "training_complete_no_test.json", result)
        print(json.dumps({"event": "trainable_decision_no_test_complete", **result}), flush=True)
        return
    load_decision_state(model, best["decision_state"])
    test_dataset = make_dataset(cfg, "test")
    test_count = min(args.max_test_samples or len(test_dataset), len(test_dataset))
    test_loader = make_loader(
        Subset(test_dataset, list(range(test_count))),
        args.batch_size,
        False,
        args.num_workers,
        cfg,
        persistent_workers=False,
    )
    final_test = evaluate(model, test_loader, device)
    result = {
        "best_train_audit_epoch": best_epoch,
        "best_train_audit_joint": best_score,
        "final_test": final_test,
        "test_evaluation_count": 1,
    }
    write_json(output_dir / "final_test_metrics.json", result)
    torch.save(
        {
            **best,
            "final_test_metrics": final_test,
            "action_scales": model.action_scales.detach().cpu(),
            "reason_scales": model.reason_scales.detach().cpu(),
            "threshold_prob": model.threshold_prob.detach().cpu(),
        },
        output_dir / "checkpoint_final_trained_decision.pth",
    )
    print(json.dumps({"event": "trainable_decision_final_test", **result}), flush=True)


if __name__ == "__main__":
    main()
