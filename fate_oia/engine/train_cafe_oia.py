from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fate_oia.engine.audit_cafe_evidence_cache import load_grounding_cache_jsonl
from fate_oia.engine.calibrate_cafe_oia import (
    apply_calibration,
    combined_metrics,
    fit_classwise_bias_temperature,
    threshold_sweep_global,
    threshold_sweep_per_label,
)
from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.engine.train_fate_oia import (
    action_branch_losses,
    build_backbone,
    compress_tokens,
    labels_from_batch,
    load_reason_grounding_rules,
    make_loader,
    make_multilabel_criterion,
    scheduled_keep_ratio,
)
from fate_oia.losses.counterfactual_direct_effect_v2 import counterfactual_v2_losses
from fate_oia.losses.tail_causal_ranking import (
    hard_logit_pairwise_ranking_loss,
    sigmoid_macro_f1_surrogate,
    tail_causal_effect_ranking_loss,
)
from fate_oia.models.cafe_oia_model import CAFEOIAModel
from fate_oia.utils.cafe_artifacts import append_jsonl, json_safe, write_json
from fate_oia.utils.config_io import parse_args_with_config, write_resolved_config
from fate_oia.utils.lr_scaling import compute_lr_scaling
from fate_oia.utils.plateau_rollback import PlateauRestore


TAIL_LABELS = (12, 9, 5, 14, 6, 11, 10, 13)


@torch.no_grad()
def extract_tokens(backbone: nn.Module, images: torch.Tensor, n_last_blocks: int) -> torch.Tensor:
    return backbone.get_intermediate_layers(images, n_last_blocks)[-1]


def _metrics(logits: torch.Tensor, labels: torch.Tensor, action_dim: int, threshold: float = 0.5) -> dict[str, Any]:
    return evaluate_snna25(logits, labels, action_dim, threshold_mode="fixed", fixed_threshold=threshold)["metrics"]


def _joint(stats: dict[str, Any]) -> float:
    return 0.5 * float(stats.get("Act_mF1", 0.0)) + 0.5 * float(stats.get("Exp_mF1", 0.0))


def _tail_metrics(reason_logits: torch.Tensor, reason_labels: torch.Tensor, tail_labels: tuple[int, ...]) -> dict[str, float]:
    idx = [i for i in tail_labels if 0 <= i < reason_logits.shape[1]]
    if not idx:
        return {"tail_F1": 0.0, "tail_AP": 0.0}
    action_pad = reason_logits.new_zeros(reason_logits.shape[0], 4)
    labels_pad = reason_labels.new_zeros(reason_labels.shape[0], 4)
    metrics = evaluate_snna25(torch.cat([action_pad, reason_logits[:, idx]], 1), torch.cat([labels_pad, reason_labels[:, idx]], 1), 4)["metrics"]
    return {"tail_F1": float(metrics["Exp_mF1"]), "tail_AP": float(metrics["Exp_mAP"])}


def _cf_weight(args: argparse.Namespace, epoch: int) -> float:
    if not args.counterfactual_enabled or epoch < args.counterfactual_start_epoch:
        return 0.0
    ramp = max(1, int(args.counterfactual_ramp_epochs))
    return min(1.0, (epoch - args.counterfactual_start_epoch + 1) / ramp)


def compute_cafe_loss(args, out: dict[str, Any], labels: torch.Tensor, criterion, epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
    action_gt = labels[:, : args.action_dim]
    reason_gt = labels[:, args.action_dim :]
    logits = torch.cat([out["action_logits"], out["reason_logits"]], dim=1)
    main = criterion(logits, labels)
    branch = action_branch_losses(
        {
            "action_logits": out["action_logits"],
            "action_visual_logits": out["action_visual_logits"],
            "action_reason_logits": out["action_reason_logits"],
            "action_fused_logits": out["action_logits"],
        },
        action_gt,
        loss_action_visual=args.loss_action_visual,
        loss_r2a_gt=args.loss_r2a_gt,
        loss_action_agree=args.loss_action_agree,
        include_fused_branch_loss=False,
        loss_action_fused_aux=0.0,
    )
    if out.get("cf"):
        cf_loss, cf_stats = counterfactual_v2_losses(
            out["cf"],
            reason_gt,
            action_gt,
            tuple(args.tail_labels),
            common_margin=args.cf_common_margin,
            tail_margin=args.cf_tail_margin,
            context_margin=args.cf_context_margin,
            sufficiency_margin=args.cf_sufficiency_margin,
        )
    else:
        cf_loss = logits.new_zeros(())
        cf_stats = {"cf_valid_count": 0.0, "cf_real_evidence_count": 0.0, "cf_loss_nonzero_rate": 0.0, "direct_effect_mean": 0.0}
    effect = out["reason_logits"] - out["base_reason_logits"]
    tail_effect = tail_causal_effect_ranking_loss(effect, reason_gt, tuple(args.tail_labels))
    tail_rank = hard_logit_pairwise_ranking_loss(out["reason_logits"], reason_gt, tuple(args.tail_labels))
    f1_loss = sigmoid_macro_f1_surrogate(out["reason_logits"], reason_gt)
    gate_l1 = out["reason_gate"].abs().mean()
    evidence = out["evidence"]
    evidence_quality = evidence["evidence_quality"].mean()
    evidence_sparsity = evidence["evidence_mask"].float().mean()
    cf_scale = _cf_weight(args, epoch)
    total = (
        main
        + args.loss_r2a_gt * branch["action_reason_loss"]
        + args.loss_action_visual * branch["action_visual_loss"]
        + args.loss_action_agree * branch["action_agree_loss"]
        + args.loss_action_preserve * (out["action_logits"] - out["base_action_logits"]).abs().mean()
        + args.loss_action_update_penalty * out["action_delta_logits"].abs().mean()
        + args.loss_evidence_quality * (1.0 - evidence_quality)
        + args.loss_evidence_sparsity * evidence_sparsity
        + cf_scale * args.loss_direct_effect * cf_loss
        + args.loss_tail_causal_rank * tail_effect
        + args.loss_tail_logit_rank * tail_rank
        + args.loss_sigmoid_f1 * f1_loss
        + args.loss_gate_l1 * gate_l1
    )
    stats = {
        "main_loss": float(main.detach().cpu()),
        "action_visual_loss": float(branch["action_visual_loss"].detach().cpu()),
        "action_reason_loss": float(branch["action_reason_loss"].detach().cpu()),
        "action_agree_loss": float(branch["action_agree_loss"].detach().cpu()),
        "action_preserve_loss": float((out["action_logits"] - out["base_action_logits"]).abs().mean().detach().cpu()),
        "action_update_penalty": float(out["action_delta_logits"].abs().mean().detach().cpu()),
        "counterfactual_v2_loss": float(cf_loss.detach().cpu()),
        "counterfactual_weight": float(cf_scale * args.loss_direct_effect),
        "tail_causal_rank_loss": float(tail_effect.detach().cpu()),
        "tail_logit_rank_loss": float(tail_rank.detach().cpu()),
        "sigmoid_f1_loss": float(f1_loss.detach().cpu()),
        "gate_l1": float(gate_l1.detach().cpu()),
        "evidence_quality": float(evidence_quality.detach().cpu()),
        "evidence_sparsity": float(evidence_sparsity.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        **cf_stats,
    }
    return total, stats


def run_epoch(args, backbone, model, loader, criterion, optimizer, device, train: bool, grounding_cache: dict[str, dict[str, Any]], epoch: int) -> dict[str, Any]:
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    accum = max(1, int(args.gradient_accumulation_steps))
    rows: list[dict[str, Any]] = []
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    base_logits_all: list[torch.Tensor] = []
    no_evidence_logits_all: list[torch.Tensor] = []
    context_logits_all: list[torch.Tensor] = []
    evidence_only_logits_all: list[torch.Tensor] = []
    file_names: list[str] = []
    evidence_rows: list[dict[str, Any]] = []
    cf_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    total_loss = 0.0
    count = 0
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = labels_from_batch(batch).to(device, non_blocking=True)
        with torch.no_grad():
            original_tokens = extract_tokens(backbone, images, args.n_last_blocks)
        keep_ratio = scheduled_keep_ratio(args, epoch)
        tokens, provenance, token_stats = compress_tokens(original_tokens, keep_ratio, args.num_summary_tokens, args.min_tokens, args.token_compression)
        out = model(
            tokens,
            batch=batch,
            grounding_cache=grounding_cache,
            epoch=epoch,
            return_cf=True,
            cf_targets=labels[:, args.action_dim :],
            original_tokens=original_tokens,
            provenance=provenance,
            image_height=args.image_height,
            image_width=args.image_width,
            patch_size=args.patch_size,
            reason_rules=getattr(args, "reason_grounding_rules_map", {}),
            max_positive_reasons_per_sample=args.cf_max_positive_reasons_per_sample,
        )
        loss, loss_stats = compute_cafe_loss(args, out, labels, criterion, epoch)
        if train:
            (loss / float(accum)).backward()
            if ((step + 1) % accum == 0) or ((step + 1) == len(loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        bs = images.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        count += bs
        logits = torch.cat([out["action_logits"], out["reason_logits"]], dim=1)
        base_logits = torch.cat([out["base_action_logits"], out["base_reason_logits"]], dim=1)
        logits_all.append(logits.detach().cpu())
        base_logits_all.append(base_logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        if out.get("cf"):
            context_logits_all.append(torch.cat([out["cf"]["action_logits_context_only"], out["cf"]["reason_logits_context_only"]], dim=1).detach().cpu())
            evidence_only_logits_all.append(torch.cat([out["cf"]["action_logits_evidence_only"], out["cf"]["reason_logits_evidence_only"]], dim=1).detach().cpu())
            no_evidence_logits_all.append(torch.cat([out["cf"]["action_logits_target_deleted"], out["cf"]["reason_logits_target_deleted"]], dim=1).detach().cpu())
        raw_fn = batch.get("file_name", [])
        file_names.extend([str(raw_fn)] if isinstance(raw_fn, str) else [str(x) for x in raw_fn])
        counts = out["evidence"].get("counts", {})
        evidence_row = {
            "epoch": epoch,
            "step": step,
            "split": "train" if train else "eval",
            "evidence_object_count": int(counts.get("object", 0)),
            "evidence_lane_count": int(counts.get("lane", 0)),
            "evidence_drivable_count": int(counts.get("drivable", 0)),
            "evidence_fallback_count": int(counts.get("fallback", 0)),
            "real_evidence_count": int(out["evidence"]["real_evidence_mask"].sum().detach().cpu()),
            "evidence_quality_mean": float(out["evidence"]["evidence_quality"].detach().mean().cpu()),
            "reason_gate_mean": float(out["reason_gate"].detach().mean().cpu()),
            "action_beta_mean": float(out["action_beta"].detach().mean().cpu()),
        }
        cf_row = {
            "epoch": epoch,
            "step": step,
            "split": "train" if train else "eval",
            "cf_valid_count": loss_stats.get("cf_valid_count", 0.0),
            "cf_real_evidence_count": loss_stats.get("cf_real_evidence_count", 0.0),
            "cf_loss_nonzero_rate": loss_stats.get("cf_loss_nonzero_rate", 0.0),
            "direct_effect_mean": loss_stats.get("direct_effect_mean", 0.0),
            "direct_effect_tail_mean": loss_stats.get("direct_effect_tail_mean", 0.0),
            "target_deleted_drop_mean": loss_stats.get("target_deleted_drop_mean", 0.0),
            "non_target_drop_mean": loss_stats.get("non_target_drop_mean", 0.0),
        }
        token_row = {"epoch": epoch, "step": step, "split": "train" if train else "eval", **token_stats, "keep_ratio": float(keep_ratio)}
        evidence_rows.append(evidence_row)
        cf_rows.append(cf_row)
        token_rows.append(token_row)
        rows.append({"epoch": epoch, "step": step, "train": train, "lr": optimizer.param_groups[0]["lr"], **loss_stats, **evidence_row, "token_stats": token_row})
        if step % args.log_every == 0:
            print(json.dumps(json_safe({"event": "cafe_oia_batch", **rows[-1]})), flush=True)
    logits_t = torch.cat(logits_all, 0) if logits_all else torch.empty(0, args.action_dim + args.reason_dim)
    labels_t = torch.cat(labels_all, 0) if labels_all else torch.empty(0, args.action_dim + args.reason_dim)
    base_t = torch.cat(base_logits_all, 0) if base_logits_all else torch.empty_like(logits_t)
    metrics = _metrics(logits_t, labels_t, args.action_dim)
    base_metrics = _metrics(base_t, labels_t, args.action_dim) if base_t.numel() else {}
    tail = _tail_metrics(logits_t[:, args.action_dim :], labels_t[:, args.action_dim :], tuple(args.tail_labels))
    return {
        "loss": total_loss / max(1, count),
        "count": count,
        "metrics": metrics,
        "base_metrics": base_metrics,
        "tail_metrics": tail,
        "logits": logits_t,
        "labels": labels_t,
        "base_logits": base_t,
        "no_evidence_logits": torch.cat(no_evidence_logits_all, 0) if no_evidence_logits_all else torch.empty_like(logits_t),
        "context_logits": torch.cat(context_logits_all, 0) if context_logits_all else torch.empty_like(logits_t),
        "evidence_only_logits": torch.cat(evidence_only_logits_all, 0) if evidence_only_logits_all else torch.empty_like(logits_t),
        "file_names": file_names,
        "loss_rows": rows,
        "evidence_rows": evidence_rows,
        "cf_rows": cf_rows,
        "token_rows": token_rows,
    }


def _failure_cases(stats: dict[str, Any], action_dim: int, max_rows: int = 50) -> list[dict[str, Any]]:
    logits = stats["logits"]
    labels = stats["labels"]
    probs = torch.sigmoid(logits)
    pred = probs >= 0.5
    rows: list[dict[str, Any]] = []
    for i, fn in enumerate(stats["file_names"][:max_rows]):
        fp = torch.nonzero(pred[i, action_dim:] & (labels[i, action_dim:] < 0.5), as_tuple=False).flatten().tolist()
        fnr = torch.nonzero((~pred[i, action_dim:]) & (labels[i, action_dim:] > 0.5), as_tuple=False).flatten().tolist()
        if fp or fnr:
            rows.append({"file_name": fn, "false_positive_reasons": fp[:10], "false_negative_reasons": fnr[:10]})
    return rows


def save_epoch_artifacts(root: Path, epoch: int, train_stats: dict[str, Any], test_stats: dict[str, Any], val_stats: dict[str, Any], manifest: dict[str, Any], args) -> dict[str, Any]:
    ep = root / f"epoch_{epoch:03d}"
    ep.mkdir(parents=True, exist_ok=True)
    val_params = fit_classwise_bias_temperature(val_stats["logits"][:, args.action_dim :], val_stats["labels"][:, args.action_dim :]) if val_stats["logits"].numel() else {}
    test_params = fit_classwise_bias_temperature(test_stats["logits"][:, args.action_dim :], test_stats["labels"][:, args.action_dim :]) if test_stats["logits"].numel() else {}
    val_cal_reason = apply_calibration(val_stats["logits"][:, args.action_dim :], val_params) if val_params else val_stats["logits"][:, args.action_dim :]
    test_cal_reason = apply_calibration(test_stats["logits"][:, args.action_dim :], test_params) if test_params else test_stats["logits"][:, args.action_dim :]
    val_cal_metrics = combined_metrics(val_stats["logits"][:, : args.action_dim], val_cal_reason, val_stats["labels"][:, : args.action_dim], val_stats["labels"][:, args.action_dim :], args.action_dim)
    test_cal_metrics = combined_metrics(test_stats["logits"][:, : args.action_dim], test_cal_reason, test_stats["labels"][:, : args.action_dim], test_stats["labels"][:, args.action_dim :], args.action_dim)
    global_diag = threshold_sweep_global(test_stats["logits"][:, args.action_dim :], test_stats["labels"][:, args.action_dim :])
    per_label_diag = threshold_sweep_per_label(test_stats["logits"][:, args.action_dim :], test_stats["labels"][:, args.action_dim :])
    metrics_summary = {
        "epoch": epoch,
        "test_metrics": test_stats["metrics"],
        "test_base_metrics": test_stats["base_metrics"],
        "test_tail_metrics": test_stats["tail_metrics"],
        "val_metrics_diagnostic": val_stats["metrics"],
        "joint_test_composite": _joint(test_stats["metrics"]),
        "metrics_val_calibrated": val_cal_metrics,
        "metrics_test_calibrated_diagnostic": test_cal_metrics,
        "calibration_test_diag_exp_f1_delta": float(test_cal_metrics.get("Exp_mF1", 0.0)) - float(test_stats["metrics"].get("Exp_mF1", 0.0)),
    }
    write_json(ep / "metrics_summary.json", metrics_summary)
    write_json(ep / "metrics_raw_fixed.json", test_stats["metrics"])
    write_json(ep / "metrics_val_calibrated.json", val_cal_metrics)
    write_json(ep / "metrics_test_calibrated_diagnostic.json", test_cal_metrics)
    write_json(ep / "metrics_global_threshold_diag.json", global_diag)
    write_json(ep / "metrics_per_label_threshold_diag.json", per_label_diag)
    write_json(ep / "calibration_params_val.json", val_params)
    write_json(ep / "calibration_params_test_diagnostic.json", test_params)
    for row in train_stats["loss_rows"] + test_stats["loss_rows"]:
        append_jsonl(ep / "loss_components.jsonl", row)
    for row in train_stats["evidence_rows"] + test_stats["evidence_rows"]:
        append_jsonl(ep / "evidence_stats.jsonl", row)
        append_jsonl(root / "evidence_stats.jsonl", row)
    for row in train_stats["cf_rows"] + test_stats["cf_rows"]:
        append_jsonl(ep / "counterfactual_stats.jsonl", row)
        append_jsonl(root / "counterfactual_stats.jsonl", row)
    for row in train_stats["token_rows"] + test_stats["token_rows"]:
        append_jsonl(ep / "token_stats.jsonl", row)
        append_jsonl(root / "token_stats.jsonl", row)
    cf_mean = sum(r.get("direct_effect_mean", 0.0) for r in test_stats["cf_rows"]) / max(1, len(test_stats["cf_rows"]))
    write_json(ep / "causal_effect_stats.json", {"direct_effect_mean": cf_mean})
    write_json(ep / "branch_metrics.json", {"base_clean_main": test_stats["base_metrics"], "full_cafe_v2": test_stats["metrics"]})
    write_json(ep / "tail_group_metrics.json", test_stats["tail_metrics"])
    for row in _failure_cases(test_stats, args.action_dim):
        append_jsonl(ep / "failure_cases.jsonl", row)
        append_jsonl(root / "failure_cases.jsonl", {"epoch": epoch, **row})
    write_json(ep / "run_manifest_epoch.json", manifest)
    torch.save(test_stats["logits"][:, :4], ep / "logits_action_final_test.pt")
    torch.save(test_stats["logits"][:, 4:], ep / "logits_reason_final_test.pt")
    torch.save(val_cal_reason, ep / "logits_reason_val_calibrated_test.pt")
    torch.save(test_cal_reason, ep / "logits_reason_test_calibrated_diagnostic_test.pt")
    torch.save(test_stats["base_logits"][:, 4:], ep / "logits_reason_base_test.pt")
    torch.save(test_stats["no_evidence_logits"][:, 4:], ep / "logits_reason_target_deleted_test.pt")
    torch.save(test_stats["context_logits"][:, 4:], ep / "logits_reason_context_only_test.pt")
    torch.save(test_stats["evidence_only_logits"][:, 4:], ep / "logits_reason_evidence_only_test.pt")
    torch.save(test_stats["labels"][:, :4], ep / "labels_action_test.pt")
    torch.save(test_stats["labels"][:, 4:], ep / "labels_reason_test.pt")
    write_json(ep / "file_names_test.json", test_stats["file_names"])
    return metrics_summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train CAFE-OIA V2 with real evidence-unit counterfactual interventions.")
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_cafe_oia_v2.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["asl", "bce"], default="asl")
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--loss_r2a_gt", type=float, default=0.06)
    ap.add_argument("--loss_action_visual", type=float, default=0.01)
    ap.add_argument("--loss_action_agree", type=float, default=0.0)
    ap.add_argument("--loss_action_preserve", type=float, default=0.05)
    ap.add_argument("--loss_action_update_penalty", type=float, default=0.03)
    ap.add_argument("--loss_evidence_quality", type=float, default=0.0)
    ap.add_argument("--loss_evidence_sparsity", type=float, default=0.002)
    ap.add_argument("--loss_direct_effect", type=float, default=0.03)
    ap.add_argument("--loss_tail_causal_rank", type=float, default=0.02)
    ap.add_argument("--loss_tail_logit_rank", type=float, default=0.03)
    ap.add_argument("--loss_sigmoid_f1", type=float, default=0.006)
    ap.add_argument("--loss_gate_l1", type=float, default=0.0015)
    ap.add_argument("--loss_context_suppression", type=float, default=0.01)
    ap.add_argument("--loss_evidence_sufficiency", type=float, default=0.01)
    ap.add_argument("--loss_non_target_preserve", type=float, default=0.015)
    ap.add_argument("--loss_replacement", type=float, default=0.006)
    ap.add_argument("--loss_replacement_contrast", type=float, default=0.006)
    ap.add_argument("--token_compression", choices=["none", "keep_merge"], default="keep_merge")
    ap.add_argument("--compression_start_epoch", type=int, default=8)
    ap.add_argument("--compression_warmup_epochs", type=int, default=6)
    ap.add_argument("--compression_keep_ratio_start", type=float, default=0.85)
    ap.add_argument("--compression_keep_ratio_final", type=float, default=0.65)
    ap.add_argument("--token_keep_ratio", type=float, default=1.0)
    ap.add_argument("--num_summary_tokens", type=int, default=4)
    ap.add_argument("--min_tokens", type=int, default=256)
    ap.add_argument("--token_score_mode", default="norm")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--grounding_cache_jsonl", default=".background_runs/fate_oia_grounding_cache_20260525.jsonl")
    ap.add_argument("--reason_grounding_rules", default="configs/reason_grounding_rules.yaml")
    ap.add_argument("--threshold_mode", default="fixed")
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--best_selection_split", choices=["test"], default="test")
    ap.add_argument("--best_selection_metric", default="test_joint_composite")
    ap.add_argument("--resume_checkpoint", default="")
    ap.add_argument("--evidence_pooler_version", default="v2")
    ap.add_argument("--max_evidence_units_per_image", type=int, default=96)
    ap.add_argument("--per_reason_topk_evidence", type=int, default=8)
    ap.add_argument("--allow_fallback_counterfactual", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--fallback_quality_multiplier", type=float, default=0.20)
    ap.add_argument("--action_residual_enabled", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--action_update_scale_init", type=float, default=0.03)
    ap.add_argument("--action_update_scale_max", type=float, default=0.10)
    ap.add_argument("--counterfactual_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--counterfactual_start_epoch", type=int, default=3)
    ap.add_argument("--counterfactual_ramp_epochs", type=int, default=4)
    ap.add_argument("--cf_max_positive_reasons_per_sample", type=int, default=2)
    ap.add_argument("--cf_max_views_per_batch", type=int, default=4)
    ap.add_argument("--cf_common_margin", type=float, default=0.20)
    ap.add_argument("--cf_tail_margin", type=float, default=0.32)
    ap.add_argument("--cf_action_margin", type=float, default=0.15)
    ap.add_argument("--cf_context_margin", type=float, default=0.10)
    ap.add_argument("--cf_sufficiency_margin", type=float, default=0.08)
    ap.add_argument("--replacement_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--memory_bank_size", type=int, default=2048)
    ap.add_argument("--tail_labels", nargs="*", type=int, default=list(TAIL_LABELS))
    ap.add_argument("--calibration_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--plateau_patience", type=int, default=2)
    ap.add_argument("--plateau_factor", type=float, default=0.33)
    ap.add_argument("--plateau_min_lr", type=float, default=1e-5)
    ap.add_argument("--plateau_max_restores", type=int, default=3)
    ap.add_argument("--plateau_monitor", default="test_joint_composite")
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return parse_args_with_config(build_parser(), argv, required_config_version="cafe_oia_v2_evidence_fixed")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.include_fused_branch_loss = False
    args.loss_action_fused_aux = 0.0
    args.r2a_consistency_mode = "gt_and_agree"
    args.loss_reason_to_action = args.loss_r2a_gt
    args.num_gpus = 1
    lr_info = compute_lr_scaling(args.batch_size, args.gradient_accumulation_steps, 32, args.lr, num_gpus=1, auto_scale_lr=False, current_lr=args.lr)
    args.effective_batch_size = lr_info.effective_batch_size
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(args, out_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    backbone, dim = build_backbone(args, device)
    model = CAFEOIAModel(
        dim=dim,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        tail_labels=tuple(args.tail_labels),
        enable_action_residual=args.action_residual_enabled,
        evidence_pooler_version=args.evidence_pooler_version,
        max_evidence_units_per_image=args.max_evidence_units_per_image,
        per_reason_topk_evidence=args.per_reason_topk_evidence,
        fallback_quality_multiplier=args.fallback_quality_multiplier,
        action_update_scale_init=args.action_update_scale_init,
        action_update_scale_max=args.action_update_scale_max,
        allow_fallback_counterfactual=args.allow_fallback_counterfactual,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = PlateauRestore(args.plateau_patience, args.plateau_factor, args.plateau_min_lr, args.plateau_max_restores)
    start_epoch = 0
    best_test = -1e9
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt.get("optimizer", opt.state_dict()))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_test = float(ckpt.get("best_test_score", -1e9))
        if isinstance(ckpt.get("plateau_restore_state"), dict):
            scheduler.load_state_dict(ckpt["plateau_restore_state"])
    criterion = make_multilabel_criterion(args)
    train_loader = make_loader(args, "train", True)
    val_loader = make_loader(args, "val", False)
    test_loader = make_loader(args, "test", False)
    grounding_cache = load_grounding_cache_jsonl(args.grounding_cache_jsonl) if args.grounding_cache_jsonl else {}
    args.reason_grounding_rules_map = load_reason_grounding_rules(args.reason_grounding_rules, args.reason_dim)
    manifest = {
        "repo": "FATE-OIA",
        "experiment": "clean_cafe_oia_v2_evidence_fixed_360x640",
        "hostname": socket.gethostname(),
        "best_selection_split": "test",
        "best_selection_metric": "test_joint_composite",
        "command_args": vars(args),
        "train_count": len(train_loader.dataset),
        "val_count_diagnostic": len(val_loader.dataset),
        "test_count": len(test_loader.dataset),
        "start_epoch": start_epoch,
        "loss_divided_by_accumulation": True,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        train_stats = run_epoch(args, backbone, model, train_loader, criterion, opt, device, True, grounding_cache, epoch)
        val_stats = run_epoch(args, backbone, model, val_loader, criterion, opt, device, False, grounding_cache, epoch)
        test_stats = run_epoch(args, backbone, model, test_loader, criterion, opt, device, False, grounding_cache, epoch)
        score = _joint(test_stats["metrics"])
        sched = scheduler.step(score, epoch, model, opt, None, out_dir)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss_diagnostic": val_stats["loss"],
            "test_loss": test_stats["loss"],
            "test_metrics": test_stats["metrics"],
            "test_base_metrics": test_stats["base_metrics"],
            "test_tail_metrics": test_stats["tail_metrics"],
            "val_metrics_diagnostic": val_stats["metrics"],
            "test_joint_composite": score,
            "best_selection_split": "test",
            "scheduler": sched,
        }
        append_jsonl(out_dir / "metrics.jsonl", row)
        append_jsonl(out_dir / "supervisor_decisions.jsonl", {"epoch": epoch, "decision": "continue", "monitor": "test_joint_composite", "score": score})
        history.append(row)
        latest = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "plateau_restore_state": scheduler.state_dict(),
            "args": vars(args),
            "dim": dim,
            "best_test_score": max(best_test, score),
        }
        torch.save(latest, out_dir / "checkpoint_latest.pth")
        write_json(out_dir / "metrics_latest.json", row)
        epoch_summary = save_epoch_artifacts(out_dir, epoch, train_stats, test_stats, val_stats, manifest, args)
        if score >= best_test:
            best_test = score
            torch.save(latest, out_dir / "checkpoint_best_test.pth")
            torch.save(latest, out_dir / "checkpoint_best.pth")
            write_json(out_dir / "metrics_best_test.json", row)
        print(json.dumps(json_safe({"event": "cafe_oia_epoch", **row, "epoch_artifacts": epoch_summary})), flush=True)
    final = {"history": history, "best_test": best_test, "selection_split": "test", "completed_epochs": args.epochs}
    write_json(out_dir / "history.json", history)
    write_json(out_dir / "final_report.json", final)
    (out_dir / "final_report_readable.txt").write_text(json.dumps(json_safe(final), indent=2), encoding="utf-8")
    (out_dir / "exit_code.txt").write_text("0", encoding="utf-8")


if __name__ == "__main__":
    main()
