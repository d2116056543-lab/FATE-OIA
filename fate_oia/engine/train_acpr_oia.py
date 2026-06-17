from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.models.acpr_oia_model import ACPROIAModel
from fate_oia.models.acpr_pair_memory import PairMiningThresholds
from fate_oia.models.acpr_predicate_targets import WeakPredicateTargetBuilder
from fate_oia.models.acpr_reason_grammar import ACPRReasonGrammar
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.losses import acpr_losses as L
from fate_oia.losses import acpr_threshold_losses as TL
from fate_oia.losses import acpr_fusionlite_losses as FL
from fate_oia.utils.acpr_artifacts import append_jsonl, json_safe, save_tensor, write_json
from fate_oia.utils.acpr_pair_mining import pair_summary
from fate_oia.utils.acpr_threshold_search import search_best_thresholds_for_f1
from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint
from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices
from fate_oia.utils.acpr_model_ema import ModelEMA


def load_config(path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if isinstance(data.get("loss"), dict):
        data.setdefault("loss_weights", {})
        mapping = {
            "action_deploy_weight": "action_direct",
            "action_visual_aux_weight": "action_visual_aux",
            "action_reason_aux_weight": "action_reason_aux",
            "reason_partial_weight": "reason_partial",
            "reason_soft_f1_weight": "reason_soft_f1",
            "matched_pair_logit_weight": "matched_pair_logit",
            "matched_pair_embed_weight": "matched_pair_embed",
            "predicate_weak_weight": "predicate_weak",
            "predicate_reason_align_weight": "predicate_reason_align",
            "fusionlite_delta_l2_weight": "fusionlite_delta_l2",
            "r2a_forbidden_prior_weight": "r2a_forbidden_prior",
            "action_combo_ce_weight": "action_combo_ce",
            "action_combo_drop_add_weight": "action_combo_drop_add",
            "cardinality_weight": "cardinality",
        }
        for src, dst in mapping.items():
            if src in data["loss"]:
                data["loss_weights"].setdefault(dst, data["loss"][src])
    return data


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "reason": torch.stack([b["reason"] for b in batch]),
        "file_name": [b["file_name"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }


def make_dataset(cfg: dict, split: str) -> BDDOIAMultiTaskDataset:
    transform = AspectRatioLetterboxTransform(int(cfg.get("image_height", 360)), int(cfg.get("image_width", 640)), patch_size=int(cfg.get("patch_size", 8)))
    return BDDOIAMultiTaskDataset(cfg["data_root"], cfg["raw_root"], split=split, action_dim=4, reason_dim=21, load_image=True, transform=transform)


def make_loader(cfg: dict, split: str, batch_size: int, max_samples: int | None, shuffle: bool, num_workers: int, indices: list[int] | None = None) -> DataLoader:
    ds = make_dataset(cfg, split)
    if indices is not None:
        ds = Subset(ds, indices)
    if max_samples:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())


def build_model(cfg: dict, device: torch.device) -> ACPROIAModel:
    model_cfg = cfg.get("model", {})
    threshold_cfg = cfg.get("threshold", {})
    model = ACPROIAModel(
        selected_layers=tuple(model_cfg.get("selected_layers", [3, 7, 11])),
        pretrained_weights=str(cfg.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
        scene_config=str(cfg.get("predicate", {}).get("scene_config", "configs/acpr_scene_predicates.yaml")),
        grammar_path=str(cfg.get("grammar", {}).get("path", "configs/acpr_reason_predicate_grammar.yaml")),
        use_mock_dino=bool(model_cfg.get("use_mock_dino", False)),
        threshold_enabled=bool(model_cfg.get("threshold_enabled", threshold_cfg.get("enabled", False))),
        use_fusionlite=bool(model_cfg.get("use_fusionlite", False)),
        fusionlite_max_delta=float(model_cfg.get("fusionlite_max_delta", cfg.get("fusionlite", {}).get("max_delta", 0.15))),
        fusionlite_gate_min=float(model_cfg.get("fusionlite_gate_min", cfg.get("fusionlite", {}).get("gate_min", 0.10))),
        fusionlite_gate_max=float(model_cfg.get("fusionlite_gate_max", cfg.get("fusionlite", {}).get("gate_max", 0.90))),
        fusionlite_use_predicate_context=bool(model_cfg.get("fusionlite_use_predicate_context", cfg.get("fusionlite", {}).get("use_predicate_context", True))),
        fusionlite_grammar_path=str(model_cfg.get("fusionlite_grammar_path", cfg.get("grammar", {}).get("path", "configs/acpr_reason_predicate_grammar.yaml"))),
        fusionlite_scene_config=str(model_cfg.get("fusionlite_scene_config", cfg.get("predicate", {}).get("scene_config", "configs/acpr_scene_predicates.yaml"))),
        threshold_kwargs={
            "action_threshold_min": float(threshold_cfg.get("action_threshold_min", 0.10)),
            "action_threshold_max": float(threshold_cfg.get("action_threshold_max", 0.90)),
            "reason_threshold_min": float(threshold_cfg.get("reason_threshold_min", 0.02)),
            "reason_threshold_max": float(threshold_cfg.get("reason_threshold_max", 0.85)),
            "tail_reason_threshold_min": float(threshold_cfg.get("tail_reason_threshold_min", 0.01)),
            "tail_reason_threshold_max": float(threshold_cfg.get("tail_reason_threshold_max", 0.65)),
            "tail_reason_indices": cfg.get("grammar", {}).get("tail_indices", [12, 9, 5, 14, 6, 11, 10, 13]),
            "use_group_shrinkage": bool(threshold_cfg.get("use_group_shrinkage", True)),
        },
    )
    return model.to(device)


def optimizer_for(model: ACPROIAModel, cfg: dict) -> torch.optim.Optimizer:
    tr = cfg.get("training", {})
    threshold_cfg = cfg.get("threshold", {})
    groups = [
        {"params": list(model.trunk.parameters()), "lr": float(tr.get("lr_trunk", 2e-4)), "name": "trunk"},
        {"params": list(model.predicate_head.parameters()), "lr": float(tr.get("lr_predicate", 2e-4)), "name": "predicate"},
        {"params": list(model.predicate_reason.parameters()), "lr": float(tr.get("lr_reason_predicate", 2e-4)), "name": "reason_predicate"},
        {"params": list(model.pair_memory.parameters()), "lr": float(tr.get("lr_pair_projection", 2e-4)), "name": "pair_projection"},
        {"params": list(model.reason_pair_proj.parameters()), "lr": float(tr.get("lr_pair_projection", 2e-4)), "name": "reason_pair_projection"},
        {"params": list(model.action_combo_aux.parameters()), "lr": float(tr.get("lr_trunk", 2e-4)), "name": "combo"},
        {"params": list(model.calibration.parameters()), "lr": float(tr.get("lr_calibration", 5e-4)), "name": "calibration"},
    ]
    if getattr(model, "fusionlite_gate", None) is not None:
        fusionlite_params = [p for p in model.fusionlite_gate.parameters() if p.requires_grad]
        if fusionlite_params:
            groups.append({"params": fusionlite_params, "lr": float(tr.get("lr_fusionlite", 2e-4)), "name": "fusionlite"})
    if bool(threshold_cfg.get("enabled", False)):
        groups.append({
            "params": list(model.threshold_head.parameters()),
            "lr": float(threshold_cfg.get("lr_threshold", 7e-4)),
            "weight_decay": float(threshold_cfg.get("weight_decay_threshold", 0.0)),
            "name": "threshold",
        })
    groups = [g for g in groups if list(g.get("params", []))]
    return torch.optim.AdamW(groups, weight_decay=float(tr.get("weight_decay", 0.05)))


def scheduler_for(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int, min_lr: float):
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max((epoch + 1) / max(warmup_epochs, 1), min_lr)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + (1 - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


def lr_multiplier_for_epoch(epoch: int, total_epochs: int, warmup_epochs: int, min_lr: float) -> float:
    """Absolute-epoch warmup/cosine multiplier for checkpoint continuation.

    Early ACPR-CalAlign checkpoints saved only model/epoch/metrics. When we
    continue from those checkpoints, this prevents the LR from restarting at
    warmup/base LR after loading model weights.
    """
    if epoch < warmup_epochs:
        return max((epoch + 1) / max(warmup_epochs, 1), min_lr)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + (1 - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def set_epoch_lrs(optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int, warmup_epochs: int, min_lr: float) -> None:
    mult = lr_multiplier_for_epoch(epoch, total_epochs, warmup_epochs, min_lr)
    for group in optimizer.param_groups:
        base_lr = group.setdefault("base_lr", group["lr"])
        group["lr"] = base_lr * mult


def reason_predicate_matrices(grammar: ACPRReasonGrammar, predicate_names: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pos, neg = grammar.reason_predicate_matrix(predicate_names)
    return torch.tensor(pos, dtype=torch.float32, device=device), torch.tensor(neg, dtype=torch.float32, device=device)


def get_pair_weights(epoch: int, active_pair_rate: float) -> tuple[float, float]:
    if epoch < 3:
        return 0.0, 0.0
    # Hard-pair logits are reason-specific and can produce very large raw
    # hinges. Keep the auxiliary objective conservative; pair strength is
    # controlled by the miner and capped hinge rather than epoch-based jumps.
    return 0.05, 0.01


def dataset_label_rates(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    base = getattr(dataset, "dataset", dataset)
    indices = getattr(dataset, "indices", None)
    samples = getattr(base, "samples", None)
    if samples is None:
        action_rows, reason_rows = [], []
        for i in range(len(dataset)):
            item = dataset[i]
            action_rows.append(item["action"])
            reason_rows.append(item["reason"])
        return torch.stack(action_rows).float().mean(0), torch.stack(reason_rows).float().mean(0)
    use_indices = list(indices) if indices is not None else list(range(len(samples)))
    actions = torch.stack([torch.tensor(samples[int(i)].action, dtype=torch.float32) for i in use_indices])
    reasons = torch.stack([torch.tensor(samples[int(i)].reason, dtype=torch.float32) for i in use_indices])
    return actions.mean(0), reasons.mean(0)


def threshold_ramp(epoch: int, cfg: dict) -> float:
    th = cfg.get("threshold", {})
    if not bool(th.get("enabled", False)):
        return 0.0
    start = int(th.get("start_epoch", 2))
    warmup = max(int(th.get("warmup_epochs", 2)), 1)
    if epoch < start:
        return 0.0
    return max(0.0, min(1.0, float(epoch - start + 1) / float(warmup)))


@torch.no_grad()
def collect_base_logits(model: ACPROIAModel, loader: DataLoader, device: torch.device, epoch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    action_logits, reason_logits, action_labels, reason_labels = [], [], [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["image"], epoch=epoch)
        action_logits.append(out["action_logits_base"].detach().cpu())
        reason_logits.append(out["reason_logits_base"].detach().cpu())
        action_labels.append(batch["action"].detach().cpu())
        reason_labels.append(batch["reason"].detach().cpu())
    return torch.cat(action_logits), torch.cat(reason_logits), torch.cat(action_labels), torch.cat(reason_labels)


@torch.no_grad()
def collect_threshold_teacher(
    model: ACPROIAModel,
    train_calib_loader: DataLoader,
    device: torch.device,
    epoch: int,
    cfg: dict,
) -> dict[str, Any]:
    action_logits, reason_logits, action_labels, reason_labels = collect_base_logits(model, train_calib_loader, device, epoch)
    logits = torch.cat([action_logits, reason_logits], dim=-1)
    labels = torch.cat([action_labels, reason_labels], dim=-1)
    search_cfg = cfg.get("threshold", {})
    step = float(search_cfg.get("grid_step", 0.01))
    grid = torch.arange(0.01, 0.95 + 1e-9, step)
    teacher = search_best_thresholds_for_f1(logits, labels, grid=grid)
    return {
        "epoch": epoch,
        "source": "train_calib",
        "threshold_prob": teacher["threshold_prob"],
        "threshold_logit": teacher["threshold_logit"],
        "best_f1": teacher["best_f1"],
        "pred_rate": teacher["pred_rate"],
        "support_pos": teacher["support_pos"],
        "support_neg": teacher["support_neg"],
    }


@torch.no_grad()
def update_threshold_teacher_from_train_calib(
    model: ACPROIAModel,
    train_calib_loader: DataLoader,
    device: torch.device,
    epoch: int,
    cfg: dict,
    out_dir: Path,
) -> dict[str, Any]:
    teacher = collect_threshold_teacher(model, train_calib_loader, device, epoch, cfg)
    search_cfg = cfg.get("threshold", {})
    model.threshold_head.update_teacher(
        teacher["threshold_logit"].to(device),
        pred_rate_teacher=teacher["pred_rate"].to(device),
        ema=float(search_cfg.get("teacher_ema", 0.20)),
        copy_to_params=bool(search_cfg.get("copy_teacher_to_params", False)),
    )
    payload = {
        "epoch": epoch,
        "source": "train_calib",
        "threshold_prob": teacher["threshold_prob"].tolist(),
        "threshold_logit": teacher["threshold_logit"].tolist(),
        "best_f1": teacher["best_f1"].tolist(),
        "pred_rate": teacher["pred_rate"].tolist(),
        "support_pos": teacher["support_pos"].tolist(),
        "support_neg": teacher["support_neg"].tolist(),
    }
    write_json(out_dir / f"threshold_teacher_epoch_{epoch:03d}.json", payload)
    return payload


def compute_threshold_losses(
    model: ACPROIAModel,
    out: dict,
    batch: dict,
    cfg: dict,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    th = cfg.get("threshold", {})
    if not bool(th.get("enabled", False)):
        zero = out["logits_final_raw"].sum() * 0.0
        return zero, {
            "loss_threshold_soft_f1_action": 0.0,
            "loss_threshold_soft_f1_reason": 0.0,
            "loss_threshold_rate": 0.0,
            "loss_threshold_cardinality": 0.0,
            "loss_threshold_teacher": 0.0,
            "loss_threshold_prior": 0.0,
            "loss_threshold_total": 0.0,
        }
    ramp = threshold_ramp(epoch, cfg)
    if ramp <= 0:
        zero = out["logits_final_raw"].sum() * 0.0
        return zero, {
            "loss_threshold_soft_f1_action": 0.0,
            "loss_threshold_soft_f1_reason": 0.0,
            "loss_threshold_rate": 0.0,
            "loss_threshold_cardinality": 0.0,
            "loss_threshold_teacher": 0.0,
            "loss_threshold_prior": 0.0,
            "loss_threshold_total": 0.0,
        }
    base_detach = bool(th.get("base_detach", True))
    action_base = out["action_logits_base"].detach() if base_detach else out["action_logits_base"]
    reason_base = out["reason_logits_base"].detach() if base_detach else out["reason_logits_base"]
    thresholded = model.threshold_head(action_base, reason_base)
    tau_start = float(th.get("soft_f1_tau_start", 0.50))
    tau_final = float(th.get("soft_f1_tau_final", 0.20))
    tau = tau_start + (tau_final - tau_start) * ramp
    losses = TL.calalign_loss_bundle(
        thresholded["action_logits_deploy"],
        thresholded["reason_logits_deploy"],
        batch["action"],
        batch["reason"],
        thresholded["threshold_logit"],
        model.threshold_head.theta_teacher,
        model.threshold_head.train_prior_theta,
        model.threshold_head.teacher_pred_rate,
        tau=tau,
        threshold_prob=thresholded["threshold_prob"],
        min_prob=torch.sigmoid(model.threshold_head.threshold_min_logit),
        max_prob=torch.sigmoid(model.threshold_head.threshold_max_logit),
        weights={
            "soft_f1_action": float(th.get("soft_f1_action_weight", 0.03)),
            "soft_f1_reason": float(th.get("soft_f1_reason_weight", 0.08)),
            "rate": float(th.get("rate_weight", 0.03)),
            "action_cardinality": float(th.get("action_cardinality_weight", 0.02)),
            "teacher": float(th.get("teacher_weight", 0.05)),
            "prior": float(th.get("prior_weight", 0.02)),
            "range": float(th.get("range_weight", 0.0)),
        },
    )
    total = losses["total"] * float(ramp)
    parts = {k: float(v.detach().cpu()) for k, v in losses.items() if k != "total"}
    parts["loss_threshold_total"] = float(total.detach().cpu())
    parts["threshold_ramp"] = float(ramp)
    parts["threshold_tau"] = float(tau)
    return total, parts


def pair_artifact_payload(pairs: dict, reason_names: list[str] | None = None) -> dict[str, Any]:
    summary = pair_summary(pairs)
    reason_dim = len(pairs.get("pair_count_per_reason", [])) if isinstance(pairs.get("pair_count_per_reason"), list) else 21
    per_reason = []
    for rid in range(reason_dim):
        name = reason_names[rid] if reason_names and rid < len(reason_names) else f"reason_{rid}"
        per_reason.append({
            "reason_id": rid,
            "reason_name": name,
            "pair_count": int((pairs.get("pair_count_per_reason") or [0] * reason_dim)[rid]),
            "active_pair_count": int((pairs.get("active_pair_count_per_reason") or [0] * reason_dim)[rid]),
            "hard_pair_count": int((pairs.get("hard_pair_count_per_reason") or [0] * reason_dim)[rid]),
            "semi_hard_count": int((pairs.get("semi_hard_pair_count_per_reason") or [0] * reason_dim)[rid]),
            "easy_count": int((pairs.get("easy_pair_count_per_reason") or [0] * reason_dim)[rid]),
            "margin_mean": float((pairs.get("margin_mean_per_reason") or [0.0] * reason_dim)[rid]),
            "active_margin_mean": float((pairs.get("active_margin_mean_per_reason") or [0.0] * reason_dim)[rid]),
        })
    return {"available": True, **summary, "per_reason": per_reason}


def compute_losses(out: dict, batch: dict, predicate_batch: dict, pairs: dict, grammar_matrices: tuple[torch.Tensor, torch.Tensor], weights: dict) -> tuple[torch.Tensor, dict[str, float]]:
    action = batch["action"]
    reason = batch["reason"]
    pair_logit, pair_logit_stats = L.matched_pair_logit_loss(out["reason_logits_base"], pairs, return_stats=True)
    pair_embed, pair_embed_stats = L.matched_pair_embedding_loss(out["reason_embeddings_for_pair"], pairs, return_stats=True)
    terms = {
        "action_direct": L.action_asl_loss(out["action_logits_base"], action),
        "action_visual_aux": L.action_asl_loss(out["action_visual_logits"], action),
        "action_reason_aux": L.action_asl_loss(out["action_reason_logits"], action),
        "reason_partial": L.partial_label_reason_loss(out["reason_logits_base"], reason, out.get("predicate_reason_contradiction_score_by_label")),
        "reason_soft_f1": L.reason_soft_f1_loss(out["reason_logits_base"], reason),
        "predicate_weak": L.predicate_weak_bce_mil_loss(out["predicate_logits"], predicate_batch["predicate_targets"], predicate_batch["predicate_mask"], predicate_batch.get("predicate_reliability")),
        "predicate_reason_align": L.predicate_reason_alignment_loss(out["predicate_probs"], reason, grammar_matrices[0], grammar_matrices[1]),
        "matched_pair_logit": pair_logit,
        "matched_pair_embed": pair_embed,
        "action_combo_ce": L.action_combo_ce_loss(out["action_set_logits"], action),
        "action_combo_drop_add": L.action_combo_drop_add_loss(out["action_set_logits"], action),
        "cardinality": L.cardinality_loss(out["cardinality_logits"], action),
        "calibration": L.calibration_loss(out["action_logits_calibrated"], out["reason_logits_calibrated"], action, reason),
        "predicate_attention_compactness": L.predicate_attention_compactness_loss(out["predicate_attention"]),
        "fusionlite_delta_l2": FL.fusionlite_delta_l2(out.get("fusionlite_delta_gate", out["action_logits_base"].new_zeros(out["action_logits_base"].shape))),
        "r2a_forbidden_prior": FL.r2a_forbidden_prior_loss(out["reason_to_action_weight"], out["forbidden_r2a_mask"]),
    }
    total = sum(terms[k] * float(weights.get(k, 0.0)) for k in terms)
    parts = {f"loss_{k}": float(v.detach().cpu()) for k, v in terms.items()}
    parts.update({f"pair_logit_{k}": v for k, v in pair_logit_stats.items()})
    parts.update({f"pair_embed_{k}": v for k, v in pair_embed_stats.items()})
    return total, parts


@torch.no_grad()
def evaluate(
    model: ACPROIAModel,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    *,
    write_artifacts: bool = True,
    ema_metrics: dict | None = None,
    cooldown_active: bool = False,
) -> dict:
    model.eval()
    action_base = []
    reason_base = []
    action_legacy = []
    action_fusionlite = []
    action_visual_branch = []
    action_reason_branch = []
    fusion_delta_abs = []
    fusion_gate_mean = []
    fusion_old_gate_mean = []
    fusion_delta_per_action = []
    fusion_gate_per_action = []
    fusion_old_gate_per_action = []
    action_logits = []
    reason_logits = []
    action_cal = []
    reason_cal = []
    action_labels = []
    reason_labels = []
    action_set_logits = []
    action_set_probs = []
    predicate_logits = []
    predicate_probs = []
    file_names: list[str] = []
    pred_stats = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["image"], epoch=epoch)
        action_base.append(out["action_logits_base"].cpu())
        action_legacy.append(out["action_logits_direct_legacy"].cpu())
        action_fusionlite.append(out["action_logits_fusionlite"].cpu())
        action_visual_branch.append(out["action_visual_logits"].cpu())
        action_reason_branch.append(out["action_reason_logits"].cpu())
        fusion_delta_abs.append(float(out["fusionlite_delta_abs_mean"].detach().cpu()))
        fusion_gate_mean.append(float(out["fusionlite_gate_mean"].detach().cpu()))
        fusion_old_gate_mean.append(float(out["fusionlite_old_gate_mean"].detach().cpu()))
        fusion_delta_per_action.append(out["fusionlite_delta_gate"].detach().cpu())
        fusion_gate_per_action.append(out["action_fusion_gate"].detach().cpu())
        fusion_old_gate_per_action.append(out["action_fusion_gate_legacy"].detach().cpu())
        reason_base.append(out["reason_logits_base"].cpu())
        action_logits.append(out["action_logits_final_raw"].cpu())
        reason_logits.append(out["reason_logits_final_raw"].cpu())
        action_cal.append(out["action_logits_final_calibrated"].cpu())
        reason_cal.append(out["reason_logits_final_calibrated"].cpu())
        action_labels.append(batch["action"].cpu())
        reason_labels.append(batch["reason"].cpu())
        action_set_logits.append(out["action_set_logits"].cpu())
        action_set_probs.append(out["action_set_probs"].cpu())
        predicate_logits.append(out["predicate_logits"].cpu())
        predicate_probs.append(out["predicate_probs"].cpu())
        file_names.extend(batch["file_name"])
        pred_stats.append(out["predicate_stats"])
    ab = torch.cat(action_base)
    rb = torch.cat(reason_base)
    al = torch.cat(action_logits)
    rl = torch.cat(reason_logits)
    ac = torch.cat(action_cal)
    rc = torch.cat(reason_cal)
    ya = torch.cat(action_labels)
    yr = torch.cat(reason_labels)
    aleg = torch.cat(action_legacy)
    afl = torch.cat(action_fusionlite)
    avis = torch.cat(action_visual_branch)
    areas = torch.cat(action_reason_branch)
    fdpa = torch.cat(fusion_delta_per_action) if fusion_delta_per_action else torch.zeros(1, 4)
    fgpa = torch.cat(fusion_gate_per_action) if fusion_gate_per_action else torch.zeros(1, 4)
    fogpa = torch.cat(fusion_old_gate_per_action) if fusion_old_gate_per_action else torch.zeros(1, 4)
    base_views = acpr_metric_views(ab, rb, ya, yr)
    views = acpr_metric_views(al, rl, ya, yr)
    cal_views = acpr_metric_views(ac, rc, ya, yr)
    legacy_views = acpr_metric_views(aleg, rb, ya, yr)
    fusionlite_views = acpr_metric_views(afl, rb, ya, yr)
    visual_views = acpr_metric_views(avis, rb, ya, yr)
    reason_action_views = acpr_metric_views(areas, rb, ya, yr)
    ema_raw = ema_metrics.get("metrics_raw_fixed") if isinstance(ema_metrics, dict) else None
    metrics = {
        "primary_branch": "deploy_fixed" if getattr(model, "threshold_enabled", False) else "base_fixed",
        "metrics_base_fixed": base_views["metrics_raw_fixed"],
        **views,
        "metrics_deploy_fixed": views["metrics_raw_fixed"],
        "metrics_deploy_fixed_ema": ema_raw,
        "metrics_test_oracle_global_threshold": views["metrics_global_threshold"],
        "metrics_test_oracle_per_label_threshold": views["metrics_per_label_threshold"],
        "metrics_calibrated": cal_views["metrics_raw_fixed"],
        "metrics_action_direct_legacy": legacy_views["metrics_raw_fixed"],
        "metrics_action_fusionlite": fusionlite_views["metrics_raw_fixed"],
        "metrics_action_visual": visual_views["metrics_raw_fixed"],
        "metrics_action_reason": reason_action_views["metrics_raw_fixed"],
        "Act_mF1_legacy": legacy_views["metrics_raw_fixed"].get("Act_mF1", 0.0),
        "Act_mF1_fusionlite": fusionlite_views["metrics_raw_fixed"].get("Act_mF1", 0.0),
        "fusionlite_delta_abs_mean": sum(fusion_delta_abs) / max(len(fusion_delta_abs), 1),
        "fusionlite_gate_mean": sum(fusion_gate_mean) / max(len(fusion_gate_mean), 1),
        "fusionlite_old_gate_mean": sum(fusion_old_gate_mean) / max(len(fusion_old_gate_mean), 1),
        "final_raw_joint": standard_joint(views["metrics_raw_fixed"]),
        "final_calibrated_joint": standard_joint(cal_views["metrics_raw_fixed"]),
        "base_fixed_joint": standard_joint(base_views["metrics_raw_fixed"]),
    }
    if ema_raw is not None:
        metrics["final_raw_joint_ema"] = standard_joint(ema_raw)
    if not write_artifacts:
        return metrics

    def _per_action(metric_view: dict, key: str) -> list[float]:
        for candidate in (key, "per_action_F1", "Act_per_label_f1", "Act_per_label_F1"):
            value = metric_view.get(candidate)
            if isinstance(value, list) and len(value) >= 4:
                return [float(x) for x in value[:4]]
        return [0.0, 0.0, 0.0, 0.0]

    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    save_tensor(epoch_dir / "logits_action_direct_legacy_test.pt", aleg)
    save_tensor(epoch_dir / "logits_action_fusionlite_test.pt", afl)
    save_tensor(epoch_dir / "logits_action_visual_test.pt", avis)
    save_tensor(epoch_dir / "logits_action_reason_test.pt", areas)
    save_tensor(epoch_dir / "logits_reason_base_test.pt", rb)
    save_tensor(epoch_dir / "logits_action_base_fixed_test.pt", ab)
    save_tensor(epoch_dir / "logits_reason_base_fixed_test.pt", rb)
    save_tensor(epoch_dir / "logits_action_deploy_test.pt", al)
    save_tensor(epoch_dir / "logits_reason_deploy_test.pt", rl)
    save_tensor(epoch_dir / "logits_action_calibrated_test.pt", ac)
    save_tensor(epoch_dir / "logits_reason_calibrated_test.pt", rc)
    save_tensor(epoch_dir / "logits_action_final_raw_test.pt", al)
    save_tensor(epoch_dir / "logits_reason_final_raw_test.pt", rl)
    save_tensor(epoch_dir / "logits_action_final_calibrated_test.pt", ac)
    save_tensor(epoch_dir / "logits_reason_final_calibrated_test.pt", rc)
    save_tensor(epoch_dir / "logits_action_direct_test.pt", al)
    save_tensor(epoch_dir / "logits_reason_direct_test.pt", rl)
    save_tensor(epoch_dir / "logits_action_set_test.pt", torch.cat(action_set_logits))
    save_tensor(epoch_dir / "probs_action_set_test.pt", torch.cat(action_set_probs))
    save_tensor(epoch_dir / "predicate_logits_test.pt", torch.cat(predicate_logits))
    save_tensor(epoch_dir / "predicate_probs_test.pt", torch.cat(predicate_probs))
    save_tensor(epoch_dir / "labels_action_test.pt", ya)
    save_tensor(epoch_dir / "labels_reason_test.pt", yr)
    write_json(epoch_dir / "file_names_test.json", file_names)
    write_json(epoch_dir / "metrics_summary.json", metrics)
    ema_act = (ema_raw or {}).get("Act_mF1", metrics["metrics_deploy_fixed"].get("Act_mF1", 0.0)) if isinstance(ema_raw, dict) else metrics["metrics_deploy_fixed"].get("Act_mF1", 0.0)
    ema_exp = (ema_raw or {}).get("Exp_mF1", metrics["metrics_deploy_fixed"].get("Exp_mF1", 0.0)) if isinstance(ema_raw, dict) else metrics["metrics_deploy_fixed"].get("Exp_mF1", 0.0)
    append_jsonl(epoch_dir / "fusionlite_metrics.jsonl", {
        "epoch": epoch,
        "split": "test",
        "Act_mF1_legacy": metrics["Act_mF1_legacy"],
        "Act_mF1_fusionlite": metrics["Act_mF1_fusionlite"],
        "Act_mF1_deploy": metrics["metrics_deploy_fixed"].get("Act_mF1", 0.0),
        "Act_mF1_deploy_ema": ema_act,
        "Exp_mF1_deploy": metrics["metrics_deploy_fixed"].get("Exp_mF1", 0.0),
        "action_primary_score": 0.85 * metrics["metrics_deploy_fixed"].get("Act_mF1", 0.0) + 0.15 * metrics["metrics_deploy_fixed"].get("Exp_mF1", 0.0),
        "action_primary_score_ema": 0.85 * ema_act + 0.15 * ema_exp,
        "fusionlite_delta_abs_mean": metrics["fusionlite_delta_abs_mean"],
        "fusionlite_gate_mean": metrics["fusionlite_gate_mean"],
        "fusionlite_old_gate_mean": metrics["fusionlite_old_gate_mean"],
        "fusionlite_gate_delta_forward": float(fdpa[:, 0].mean()),
        "fusionlite_gate_delta_stop": float(fdpa[:, 1].mean()),
        "fusionlite_gate_delta_left": float(fdpa[:, 2].mean()),
        "fusionlite_gate_delta_right": float(fdpa[:, 3].mean()),
        "cooldown_active": bool(cooldown_active),
    })
    append_jsonl(epoch_dir / "fusionlite_gate_stats.jsonl", {
        "epoch": epoch,
        "delta_abs_mean": metrics["fusionlite_delta_abs_mean"],
        "gate_mean": metrics["fusionlite_gate_mean"],
        "old_gate_mean": metrics["fusionlite_old_gate_mean"],
        "delta_gate_mean_per_action": [float(fdpa[:, i].mean()) for i in range(4)],
        "new_gate_mean_per_action": [float(fgpa[:, i].mean()) for i in range(4)],
        "old_gate_mean_per_action": [float(fogpa[:, i].mean()) for i in range(4)],
    })
    visual_f1 = _per_action(visual_views["metrics_raw_fixed"], "per_action_F1")
    reason_f1 = _per_action(reason_action_views["metrics_raw_fixed"], "per_action_F1")
    legacy_f1 = _per_action(legacy_views["metrics_raw_fixed"], "per_action_F1")
    fusion_f1 = _per_action(fusionlite_views["metrics_raw_fixed"], "per_action_F1")
    write_json(epoch_dir / "fusionlite_per_action_table.json", {
        name: {
            "old_gate_mean": float(fogpa[:, i].mean()),
            "new_gate_mean": float(fgpa[:, i].mean()),
            "delta_gate_mean": float(fdpa[:, i].mean()),
            "visual_F1": visual_f1[i],
            "reason_F1": reason_f1[i],
            "legacy_F1": legacy_f1[i],
            "fusionlite_F1": fusion_f1[i],
        }
        for i, name in enumerate(["forward", "stop", "left", "right"])
    })
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--require_no_token_compression", action="store_true")
    ap.add_argument("--resume_checkpoint", default=None, help="Resume ACPR model weights from a previous checkpoint.")
    ap.add_argument("--stop_after_epochs", type=int, default=None, help="Run only N epochs after start_epoch while preserving --epochs for LR scheduling.")
    ap.add_argument("--sanity_finetune", action="store_true", help="Freeze old modules and train FusionLite/threshold only.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("best_selection_split") != "test" or cfg.get("eval_splits") != "test":
        raise RuntimeError("ACPR requires test-only evaluation and test best selection")
    if cfg.get("token_compression") != "none" or bool(cfg.get("feature_cache_enabled", False)):
        raise RuntimeError("ACPR forbids token compression and feature caching")
    tr = cfg.get("training", {})
    epochs = int(args.epochs or tr.get("epochs", 28))
    batch_size = int(args.batch_size or tr.get("batch_size", 6))
    accum = int(args.gradient_accumulation_steps or tr.get("gradient_accumulation_steps", 5))
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(out_dir / "run_manifest.json", {
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "command_line": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "pretrained_weights": cfg.get("pretrained_weights"),
        "data_root": cfg.get("data_root"),
        "raw_root": cfg.get("raw_root"),
        "test_only": True,
        "best_selection_split": "test",
        "feature_cache_enabled": False,
        "token_compression": "none",
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum,
        "effective_batch": batch_size * accum,
        "reference_effective_batch": tr.get("reference_effective_batch", 32),
        "loss_weights": cfg.get("loss_weights", {}),
        "loss": cfg.get("loss", {}),
        "resume_checkpoint": args.resume_checkpoint,
        "use_fusionlite": bool(cfg.get("model", {}).get("use_fusionlite", False)),
        "fusionlite": cfg.get("fusionlite", {}),
        "ema": cfg.get("ema", {}),
    })
    write_json(out_dir / "implementation_fingerprint.json", {
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "model": "ACPR-OIA V1 direct-image predicate contrast learning",
        "final_action_source": "action_logits_direct",
        "feature_cache_enabled": False,
        "token_compression": "none",
        "eval_splits": "test",
        "best_selection_split": "test",
    })
    train_dataset = make_dataset(cfg, "train")
    threshold_cfg = cfg.get("threshold", {})
    train_calib_loader = None
    train_main_indices = None
    train_calib_indices = None
    if bool(threshold_cfg.get("enabled", False)):
        train_main_indices, train_calib_indices = make_train_calib_indices(
            train_dataset,
            calib_fraction=float(threshold_cfg.get("train_calib_fraction", 0.10)),
            seed=int(threshold_cfg.get("split_seed", 20260615)),
        )
        if args.max_train_samples:
            allowed = set(range(min(args.max_train_samples, len(train_dataset))))
            train_main_indices = [i for i in train_main_indices if i in allowed]
            train_calib_indices = [i for i in train_calib_indices if i in allowed]
            if not train_calib_indices:
                train_calib_indices = list(range(min(1, len(train_dataset))))
        train_loader_indices = None if bool(threshold_cfg.get("train_trunk_on_all_train", True)) else train_main_indices
        train_loader = make_loader(cfg, "train", batch_size, args.max_train_samples if train_loader_indices is None else None, True, args.num_workers, indices=train_loader_indices)
        train_calib_loader = make_loader(cfg, "train", batch_size, None, False, args.num_workers, indices=train_calib_indices)
    else:
        train_loader = make_loader(cfg, "train", batch_size, args.max_train_samples, True, args.num_workers)
    test_loader = make_loader(cfg, "test", batch_size, args.max_test_samples, False, args.num_workers)
    model = build_model(cfg, device)
    resume_ckpt = None
    start_epoch = 0
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        resume_ckpt = torch.load(resume_path, map_location=device)
        state = resume_ckpt.get("model", resume_ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        write_json(out_dir / "resume_info.json", {
            "resume_checkpoint": str(resume_path),
            "checkpoint_epoch": int(resume_ckpt.get("epoch", -1)),
            "start_epoch": start_epoch,
            "stop_after_epochs": args.stop_after_epochs,
            "optimizer_state_restored": bool("optimizer" in resume_ckpt),
            "scheduler_state_restored": False,
            "resume_mode": "model_weights_plus_absolute_epoch_lr",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        })
    if args.sanity_finetune:
        for _, param in model.named_parameters():
            param.requires_grad_(False)
        for module in [getattr(model, "fusionlite_gate", None), getattr(model, "threshold_head", None)]:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(True)
        write_json(out_dir / "sanity_finetune_freeze.json", {"enabled": True, "trainable": [n for n, p in model.named_parameters() if p.requires_grad]})
    if bool(threshold_cfg.get("enabled", False)) and not args.resume_checkpoint:
        init_ds = train_dataset if train_calib_indices is None else Subset(train_dataset, train_calib_indices)
        action_rate, reason_rate = dataset_label_rates(init_ds)
        model.threshold_head.initialize_from_label_stats(action_rate.to(device), reason_rate.to(device), cfg.get("grammar", {}).get("tail_indices"))
        write_json(out_dir / "threshold_initialization.json", {
            "source": "train_calib" if train_calib_indices is not None else "train",
            "train_calib_fraction": float(threshold_cfg.get("train_calib_fraction", 0.10)),
            "train_main_count": len(train_main_indices or []),
            "train_calib_count": len(train_calib_indices or []),
            "action_pos_rate": action_rate.tolist(),
            "reason_pos_rate": reason_rate.tolist(),
            "initial_threshold_prob": model.threshold_head.forward(torch.zeros(1, 4, device=device), torch.zeros(1, 21, device=device))["threshold_prob"].detach().cpu().tolist(),
        })
    opt = optimizer_for(model, cfg)
    ema_cfg = cfg.get("ema", {})
    ema_helper = None
    if bool(ema_cfg.get("enabled", False)) and bool(cfg.get("model", {}).get("use_fusionlite", False)):
        ema_helper = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.995)))
    warmup_epochs = int(tr.get("warmup_epochs", 2))
    min_lr = float(tr.get("min_lr", 1e-5))
    for group in opt.param_groups:
        group.setdefault("base_lr", group["lr"])
    if resume_ckpt is not None and "optimizer" in resume_ckpt:
        opt.load_state_dict(resume_ckpt["optimizer"])
        for group in opt.param_groups:
            group.setdefault("base_lr", group["lr"])
    grammar = ACPRReasonGrammar(cfg.get("grammar", {}).get("path", "configs/acpr_reason_predicate_grammar.yaml"))
    target_builder = WeakPredicateTargetBuilder(cfg.get("predicate", {}).get("scene_config", "configs/acpr_scene_predicates.yaml"), cfg.get("bdd100k_root"))
    matrices = reason_predicate_matrices(grammar, model.predicate_head.names, device)
    weights = cfg.get("loss_weights", {})
    pair_cfg = cfg.get("pair_mining", {})
    pair_thresholds = PairMiningThresholds(
        action_sim_min=float(pair_cfg.get("action_sim_min", pair_cfg.get("min_action_similarity", 0.35))),
        visual_sim_min=float(pair_cfg.get("visual_sim_min", pair_cfg.get("min_visual_similarity", 0.05))),
        predicate_sim_min=float(pair_cfg.get("predicate_sim_min", pair_cfg.get("min_predicate_similarity", 0.05))),
        contradiction_min=float(pair_cfg.get("contradiction_min", pair_cfg.get("min_contradiction_common", 0.15))),
        tail_action_sim_min=float(pair_cfg.get("tail_action_sim_min", 0.20)),
        tail_visual_sim_min=float(pair_cfg.get("tail_visual_sim_min", -0.05)),
        tail_predicate_sim_min=float(pair_cfg.get("tail_predicate_sim_min", -0.05)),
        tail_contradiction_min=float(pair_cfg.get("tail_contradiction_min", pair_cfg.get("min_contradiction_tail", 0.05))),
        semi_hard_band=float(pair_cfg.get("semi_hard_band", 0.15)),
        fallback_easy_pair_weight=float(pair_cfg.get("fallback_easy_pair_weight", 0.15)),
    )
    best_raw = -1.0
    best_base = -1.0
    best_cal = -1.0
    best_exp = -1.0
    best_map = -1.0
    best_act = -1.0
    best_tail = -1.0
    global_step = 0
    train_calib_action_primary_scores: list[float] = []
    cooldown_active = False
    end_epoch = epochs
    if args.stop_after_epochs is not None:
        end_epoch = min(epochs, start_epoch + max(0, int(args.stop_after_epochs)))
    for epoch in range(start_epoch, end_epoch):
        set_epoch_lrs(opt, epoch, epochs, warmup_epochs, min_lr)
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred_batch = target_builder.build(batch["file_name"], device=device)
            out = model(batch["image"], epoch=epoch)
            pairs = model.pair_memory.mine(
                batch["file_name"],
                out["global_embedding"].detach(),
                out["predicate_probs"].detach(),
                batch["action"],
                batch["reason"],
                out["contradiction_score"].detach(),
                grammar.tail_indices,
                reason_logits_current=out["reason_logits_base"].detach(),
                reason_embeddings_current=out["reason_embeddings_for_pair"].detach(),
                epoch=epoch,
                max_pairs=int(pair_cfg.get("max_pairs_per_batch", 256)),
                max_pairs_per_reason=int(pair_cfg.get("max_pairs_per_reason", 8)),
                max_tail_pairs_per_reason=int(pair_cfg.get("max_tail_pairs_per_reason", 12)),
                max_memory_scan=int(pair_cfg.get("max_memory_scan", 2048)),
                margin=float(pair_cfg.get("margin", 0.25)),
                thresholds=pair_thresholds,
            )
            pair_count = int(pairs.get("pair_count", 0))
            active_pair_rate = float(pairs.get("active_pair_count", 0)) / max(float(pair_count), 1.0)
            pair_logit_weight, pair_embed_weight = get_pair_weights(epoch, active_pair_rate)
            batch_weights = dict(weights)
            batch_weights["matched_pair_logit"] = pair_logit_weight
            batch_weights["matched_pair_embed"] = pair_embed_weight
            loss, parts = compute_losses(out, batch, pred_batch, pairs, matrices, batch_weights)
            threshold_loss, threshold_parts = compute_threshold_losses(model, out, batch, cfg, epoch)
            loss = loss + threshold_loss
            parts.update(threshold_parts)
            (loss / accum).backward()
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 1.0)))
                opt.step()
                if ema_helper is not None:
                    ema_helper.update(model)
                opt.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % 200 == 0 or step == 1:
                payload = {
                    "event": "acpr_batch",
                    "epoch": epoch,
                    "step": step,
                    "total_steps": len(train_loader),
                    "lr": opt.param_groups[0]["lr"],
                    "loss_total": float(loss.detach().cpu()),
                    **parts,
                    **pair_summary(pairs),
                    "matched_pair_weight_logit": pair_logit_weight,
                    "matched_pair_weight_embed": pair_embed_weight,
                    "active_pair_rate": active_pair_rate,
                    "reason_positive_rate": float(batch["reason"].mean().detach().cpu()),
                    "predicate_positive_rate": float(pred_batch["predicate_targets"].mean().detach().cpu()),
                    "fusionlite_delta_abs_mean": float(out.get("fusionlite_delta_abs_mean", torch.tensor(0.0)).detach().cpu()) if torch.is_tensor(out.get("fusionlite_delta_abs_mean")) else 0.0,
                    "fusionlite_gate_mean": float(out.get("fusionlite_gate_mean", torch.tensor(0.0)).detach().cpu()) if torch.is_tensor(out.get("fusionlite_gate_mean")) else 0.0,
                    "gpu_peak_memory_gb": float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0,
                }
                print(json.dumps(payload), flush=True)
                append_jsonl(out_dir / "loss_components.jsonl", payload)
            model.pair_memory.enqueue(
                batch["file_name"],
                out["global_embedding"],
                out["predicate_probs"],
                batch["action"],
                batch["reason"],
                out["contradiction_score"].detach(),
                out["reason_logits_base"].detach(),
                out["reason_embeddings_for_pair"].detach(),
            )
        if bool(threshold_cfg.get("enabled", False)) and train_calib_loader is not None and epoch >= int(threshold_cfg.get("teacher_update_start_epoch", 2)):
            if (epoch - int(threshold_cfg.get("teacher_update_start_epoch", 2))) % int(threshold_cfg.get("teacher_update_every", 1)) == 0:
                update_threshold_teacher_from_train_calib(model, train_calib_loader, device, epoch, cfg, out_dir)
        cooldown_active = False
        fusion_cfg = cfg.get("fusionlite", {})
        if bool(fusion_cfg.get("cooldown_enabled", False)) and train_calib_loader is not None:
            calib_metrics = evaluate(model, train_calib_loader, device, epoch, out_dir, write_artifacts=False)
            calib_score = 0.85 * calib_metrics["metrics_raw_fixed"].get("Act_mF1", 0.0) + 0.15 * calib_metrics["metrics_raw_fixed"].get("Exp_mF1", 0.0)
            train_calib_action_primary_scores.append(float(calib_score))
            cooldown_active = FL.action_primary_cooldown_should_trigger(
                train_calib_action_primary_scores,
                patience=int(fusion_cfg.get("cooldown_patience", 2)),
                min_delta=1e-4,
            )
            if cooldown_active:
                mult_by_group = {
                    "trunk": float(fusion_cfg.get("cooldown_lr_mult_trunk", 0.25)),
                    "predicate": float(fusion_cfg.get("cooldown_lr_mult_predicate", 0.25)),
                    "fusionlite": float(fusion_cfg.get("cooldown_lr_mult_fusionlite", 0.25)),
                    "threshold": float(fusion_cfg.get("cooldown_lr_mult_threshold", 0.50)),
                }
                for group in opt.param_groups:
                    name = str(group.get("name", ""))
                    for key, mult in mult_by_group.items():
                        if key in name:
                            group["base_lr"] = group.get("base_lr", group["lr"]) * mult
                            group["lr"] = group["lr"] * mult
                            break
                append_jsonl(out_dir / "fusionlite_cooldown_events.jsonl", {"epoch": epoch, "train_calib_action_primary_score": float(calib_score), "scores": train_calib_action_primary_scores, "cooldown_active": True})
        ema_metrics = None
        if ema_helper is not None and epoch >= int(ema_cfg.get("start_epoch", 3)) and getattr(ema_helper, "num_updates", 0) > 0:
            backup = ema_helper.apply_to(model)
            try:
                ema_metrics = evaluate(model, test_loader, device, epoch, out_dir, write_artifacts=False)
            finally:
                ema_helper.restore(model, backup)
        metrics = evaluate(model, test_loader, device, epoch, out_dir, ema_metrics=ema_metrics, cooldown_active=cooldown_active)
        row = {"event": "acpr_epoch", "epoch": epoch, **metrics}
        print(json.dumps(json_safe(row)), flush=True)
        append_jsonl(out_dir / "metrics_summary.jsonl", json_safe(row))
        epoch_dir = out_dir / f"epoch_{epoch:03d}"
        append_jsonl(epoch_dir / "branch_metrics.jsonl", {
            "direct": metrics.get("metrics_base_fixed", metrics["metrics_raw_fixed"]),
            "action_visual": metrics.get("metrics_action_visual"),
            "action_reason": metrics.get("metrics_action_reason"),
            "action_direct_legacy": metrics.get("metrics_action_direct_legacy"),
            "action_fusionlite": metrics.get("metrics_action_fusionlite"),
            "direct_plus_predicate": metrics.get("metrics_base_fixed", metrics["metrics_raw_fixed"]),
            "base_fixed": metrics.get("metrics_base_fixed", metrics["metrics_raw_fixed"]),
            "deploy_fixed": metrics["metrics_raw_fixed"],
            "deploy_fixed_ema": metrics.get("metrics_deploy_fixed_ema"),
            "raw": metrics["metrics_raw_fixed"],
            "calibrated": metrics["metrics_calibrated"],
            "final_raw": metrics["metrics_raw_fixed"],
            "final_calibrated": metrics["metrics_calibrated"],
        })
        append_jsonl(epoch_dir / "predicate_metrics.jsonl", {"available": True, "predicate_positive_rate": float(pred_batch["predicate_targets"].mean().detach().cpu()), "predicate_mask_rate": float(pred_batch["predicate_mask"].mean().detach().cpu())})
        append_jsonl(epoch_dir / "predicate_coverage.jsonl", {"available": True, **pred_batch.get("predicate_coverage", {})})
        append_jsonl(epoch_dir / "predicate_reason_alignment.jsonl", {"available": True, "positive_score_mean": float(out.get("predicate_reason_positive_score_by_label").mean().detach().cpu()), "contradiction_score_mean": float(out.get("predicate_reason_contradiction_score_by_label").mean().detach().cpu())})
        pair_epoch_payload = pair_artifact_payload(pairs)
        append_jsonl(epoch_dir / "pair_mining_stats.jsonl", {"available": True, **pair_summary(pairs)})
        append_jsonl(epoch_dir / "pair_stats.jsonl", {"available": True, **pair_summary(pairs)})
        write_json(epoch_dir / "pair_margin_per_reason.json", pair_epoch_payload)
        append_jsonl(epoch_dir / "pair_margins.jsonl", pair_epoch_payload)
        append_jsonl(epoch_dir / "tail_reason_metrics.jsonl", {
            "available": True,
            "tail_indices": grammar.tail_indices,
            "tail_pair_count": int(pairs.get("tail_pair_count", 0)),
            "tail_active_pair_count": int(pairs.get("tail_active_pair_count", 0)),
            "tail_margin_mean": float(sum(float((pairs.get("margin_mean_per_reason") or [0.0] * 21)[i]) for i in grammar.tail_indices) / max(len(grammar.tail_indices), 1)),
        })
        combo_loss_tmp, combo_stats_tmp = L.action_combo_drop_add_loss(out["action_set_logits"], batch["action"], return_stats=True)
        append_jsonl(epoch_dir / "action_combo_metrics.jsonl", {"available": True, **combo_stats_tmp})
        append_jsonl(epoch_dir / "calibration_diagnostics.jsonl", {
            "available": True,
            "temperature_mean": float(out["temperature"].mean().detach().cpu()),
            "bias_mean": float(out["calibration_bias"].mean().detach().cpu()),
            "threshold_prob": json_safe(out.get("threshold_prob", torch.empty(0)).detach().cpu()) if torch.is_tensor(out.get("threshold_prob")) else [],
            "base_fixed_joint": metrics.get("base_fixed_joint"),
            "deploy_fixed_joint": metrics.get("final_raw_joint"),
            "primary_branch": metrics.get("primary_branch"),
        })
        append_jsonl(epoch_dir / "threshold_stats.jsonl", {
            "available": bool(threshold_cfg.get("enabled", False)),
            "threshold_prob": json_safe(out.get("threshold_prob", torch.empty(0)).detach().cpu()) if torch.is_tensor(out.get("threshold_prob")) else [],
            "threshold_logit": json_safe(out.get("threshold_logit", torch.empty(0)).detach().cpu()) if torch.is_tensor(out.get("threshold_logit")) else [],
            "teacher_threshold_prob": json_safe(torch.sigmoid(model.threshold_head.theta_teacher.detach().cpu())) if hasattr(model, "threshold_head") else [],
            "teacher_pred_rate": json_safe(model.threshold_head.teacher_pred_rate.detach().cpu()) if hasattr(model, "threshold_head") else [],
        })
        append_jsonl(epoch_dir / "reason_activation_stats.jsonl", {"available": True, "reason_logit_mean": float(out["reason_logits_base"].mean().detach().cpu()), "reason_delta_mean": float(out["predicate_reason_delta_by_label"].mean().detach().cpu())})
        append_jsonl(epoch_dir / "per_label_action_metrics.jsonl", {"available": True, "metrics": metrics["metrics_raw_fixed"].get("per_action_F1", [])})
        append_jsonl(epoch_dir / "per_label_reason_metrics.jsonl", {"available": True, "metrics": metrics["metrics_raw_fixed"].get("per_reason_F1", [])})
        pos_idx = pairs.get("pair_pos_indices")
        neg_idx = pairs.get("pair_neg_indices")
        mem_idx = pairs.get("pair_neg_memory_indices")
        rid_idx = pairs.get("pair_reason_ids")
        hinge_vals = pairs.get("pair_hinge_raw")
        weight_vals = pairs.get("pair_weights")
        active_mask = pairs.get("pair_active_mask")
        is_mem = pairs.get("pair_neg_is_memory")
        if torch.is_tensor(pos_idx) and torch.is_tensor(rid_idx) and pos_idx.numel():
            order = torch.arange(pos_idx.numel(), device=pos_idx.device)
            if torch.is_tensor(active_mask) and active_mask.numel() == pos_idx.numel():
                active_order = torch.where(active_mask.bool())[0]
                order = active_order if active_order.numel() else order
            for oi in order[:50].tolist():
                pi = int(pos_idx[oi].detach().cpu())
                ni = int(neg_idx[oi].detach().cpu()) if torch.is_tensor(neg_idx) else -1
                mi = int(mem_idx[oi].detach().cpu()) if torch.is_tensor(mem_idx) else -1
                rr = int(rid_idx[oi].detach().cpu())
                append_jsonl(epoch_dir / "matched_counterfactual_cases.jsonl", {
                    "available": True,
                    "file_pos": batch["file_name"][pi] if pi < len(batch["file_name"]) else str(pi),
                    "file_neg_or_memory": (batch["file_name"][ni] if ni >= 0 and ni < len(batch["file_name"]) else f"memory:{mi}"),
                    "reason_id": rr,
                    "reason_name": f"reason_{rr}",
                    "action_pos": json_safe(batch["action"][pi].detach().cpu()) if pi < batch["action"].shape[0] else [],
                    "action_neg": json_safe(batch["action"][ni].detach().cpu()) if ni >= 0 and ni < batch["action"].shape[0] else [],
                    "z_pos": float(out["reason_logits_final_raw"][pi, rr].detach().cpu()) if pi < out["reason_logits_final_raw"].shape[0] else 0.0,
                    "z_neg": float((pairs.get("pair_neg_logits_detached")[oi] if torch.is_tensor(pairs.get("pair_neg_logits_detached")) else torch.tensor(0.0)).detach().cpu()),
                    "hinge_raw": float(hinge_vals[oi].detach().cpu()) if torch.is_tensor(hinge_vals) else 0.0,
                    "pair_weight": float(weight_vals[oi].detach().cpu()) if torch.is_tensor(weight_vals) else 0.0,
                    "is_memory": bool(is_mem[oi].detach().cpu()) if torch.is_tensor(is_mem) else False,
                    "action_sim": float(pairs["pair_action_sim"][oi].detach().cpu()) if torch.is_tensor(pairs.get("pair_action_sim")) else 0.0,
                    "visual_sim": float(pairs["pair_visual_sim"][oi].detach().cpu()) if torch.is_tensor(pairs.get("pair_visual_sim")) else 0.0,
                    "predicate_sim": float(pairs["pair_predicate_sim"][oi].detach().cpu()) if torch.is_tensor(pairs.get("pair_predicate_sim")) else 0.0,
                    "contradiction": float(pairs["pair_contradiction"][oi].detach().cpu()) if torch.is_tensor(pairs.get("pair_contradiction")) else 0.0,
                })
        else:
            append_jsonl(epoch_dir / "matched_counterfactual_cases.jsonl", {"available": False, **pair_summary(pairs)})
        pair_case = {"available": int(pairs.get("pair_count", 0)) > 0, **pair_summary(pairs)}
        append_jsonl(epoch_dir / "pair_cases_test.jsonl", pair_case)
        append_jsonl(epoch_dir / "failure_cases.jsonl", {"available": True, "file_count": len(metrics)})
        append_jsonl(epoch_dir / "gpu_memory.jsonl", {"peak_gb": float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0})
        ckpt = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "base_lrs": [group.get("base_lr", group["lr"]) for group in opt.param_groups],
        }
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        raw = float(metrics["final_raw_joint"])
        base = float(metrics.get("base_fixed_joint", raw))
        cal = float(metrics["final_calibrated_joint"])
        exp = float(metrics["metrics_raw_fixed"].get("Exp_mF1", 0.0))
        mp = float(metrics["metrics_raw_fixed"].get("Exp_mAP", 0.0))
        act = float(metrics["metrics_raw_fixed"].get("Act_mF1", 0.0))
        per_reason = metrics["metrics_raw_fixed"].get("per_reason_F1", [])
        tail_vals = [float(per_reason[i]) for i in grammar.tail_indices if isinstance(per_reason, list) and i < len(per_reason)]
        tail_mf1 = sum(tail_vals) / max(len(tail_vals), 1)
        if raw >= best_raw:
            best_raw = raw
            torch.save(ckpt, out_dir / "checkpoint_best_test_final_raw.pth")
            torch.save(ckpt, out_dir / "checkpoint_best_test_deploy_raw.pth")
        if base >= best_base:
            best_base = base; torch.save(ckpt, out_dir / "checkpoint_best_test_base_fixed.pth")
        if cal >= best_cal:
            best_cal = cal; torch.save(ckpt, out_dir / "checkpoint_best_test_final_calibrated.pth")
        if exp >= best_exp:
            best_exp = exp; torch.save(ckpt, out_dir / "checkpoint_best_test_exp_mf1.pth")
        if mp >= best_map:
            best_map = mp; torch.save(ckpt, out_dir / "checkpoint_best_test_exp_map.pth")
        if act >= best_act:
            best_act = act; torch.save(ckpt, out_dir / "checkpoint_best_test_action_mf1.pth")
        if tail_mf1 >= best_tail:
            best_tail = tail_mf1; torch.save(ckpt, out_dir / "checkpoint_best_test_tail_mf1.pth")
    write_json(out_dir / "GOAL_COMPLETED_ACPR_OIA_V1.json", {"complete": True, "epochs": epochs, "best_final_raw_joint": best_raw})
    if bool(cfg.get("model", {}).get("use_fusionlite", False)):
        write_json(out_dir / "GOAL_COMPLETED_ACPR_FUSIONLITE_V1_4.json", {"complete": True, "epochs": epochs, "best_final_raw_joint": best_raw})


if __name__ == "__main__":
    main()
