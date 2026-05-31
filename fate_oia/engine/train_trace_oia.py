from __future__ import annotations

import argparse
import copy
import json
import socket
import time
from pathlib import Path
from typing import Any

import torch

from fate_oia.datasets.dino_token_cache import DinoTokenCache
from fate_oia.engine.audit_cafe_evidence_cache import load_grounding_cache_jsonl
from fate_oia.engine.calibrate_cafe_oia import combined_metrics, threshold_sweep_global, threshold_sweep_per_label
from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.engine.export_trace_visuals import export_trace_case
from fate_oia.engine.train_fate_oia import build_backbone, extract_tokens, labels_from_batch, load_reason_grounding_rules, make_loader
from fate_oia.losses.action_primary_trace_losses import compute_action_primary_trace_loss
from fate_oia.losses.trace_losses import compute_trace_loss
from fate_oia.models.prototype_calibration import apply_prototype_calibration, fit_classwise_bias_temperature_reliability
from fate_oia.models.trace_oia_model import TraceOIAModel
from fate_oia.utils.action_candidate_selector import evaluate_action_candidates, select_action_candidate
from fate_oia.utils.config_io import load_yaml_config, write_resolved_config
from fate_oia.utils.trace_artifacts import append_jsonl, json_safe, required_epoch_artifacts, write_json
from fate_oia.utils.trace_optimizer_groups import build_action_primary_trace_optimizer


def _flat(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def walk(prefix: str, val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            out[prefix] = val

    walk("", cfg)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_trace_oia_v1.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="dataset/BDD-OIA")
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--grounding_cache_jsonl", default=".background_runs/fate_oia_grounding_cache_20260525.jsonl")
    ap.add_argument("--reason_grounding_rules", default="configs/reason_grounding_rules.yaml")
    ap.add_argument("--best_selection_split", choices=["test"], default="test")
    ap.add_argument("--best_selection_metric", default="test_joint_composite")
    ap.add_argument("--token_compression", choices=["none"], default="none")
    ap.add_argument("--action_final_mode", choices=["base_only", "action_safe_selector"], default="base_only")
    ap.add_argument("--lr_base_head", type=float, default=3e-4)
    ap.add_argument("--lr_transport", type=float, default=2e-4)
    ap.add_argument("--lr_action_head", type=float, default=3e-4)
    ap.add_argument("--lr_reason_head", type=float, default=2e-4)
    ap.add_argument("--lr_label_corr", type=float, default=5e-5)
    ap.add_argument("--lr_reason_alpha", type=float, default=5e-5)
    ap.add_argument("--lr_action_bias", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--loss_counterfactual_direct", type=float, default=0.025)
    ap.add_argument("--counterfactual_start_epoch", type=int, default=5)
    ap.add_argument("--cf_margin", type=float, default=0.05)
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--feature_cache_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--feature_cache_build_before_training", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--feature_cache_required_hit_rate", type=float, default=0.99)
    return ap


def parse_args(argv=None):
    ap = build_parser()
    pre, _ = ap.parse_known_args(argv)
    cfg = load_yaml_config(pre.config) if pre.config else {}
    flat = _flat(cfg)
    mapping = {
        "paths.data_root": "data_root",
        "paths.raw_root": "raw_root",
        "paths.pretrained_weights": "pretrained_weights",
        "paths.grounding_cache_jsonl": "grounding_cache_jsonl",
        "paths.reason_grounding_rules": "reason_grounding_rules",
        "model.action_dim": "action_dim",
        "model.reason_dim": "reason_dim",
        "model.image_height": "image_height",
        "model.image_width": "image_width",
        "model.patch_size": "patch_size",
        "model.arch": "arch",
        "model.token_compression": "token_compression",
        "model.action_final_mode": "action_final_mode",
        "training.epochs": "epochs",
        "training.batch_size": "batch_size",
        "training.gradient_accumulation_steps": "gradient_accumulation_steps",
        "training.num_workers": "num_workers",
        "training.device": "device",
        "training.log_every": "log_every",
        "training.best_selection_split": "best_selection_split",
        "training.best_selection_metric": "best_selection_metric",
        "training.max_train_samples": "max_train_samples",
        "training.max_test_samples": "max_test_samples",
        "optimizer.lr_base_head": "lr_base_head",
        "optimizer.lr_transport": "lr_transport",
        "optimizer.lr_action_head": "lr_action_head",
        "optimizer.lr_reason_head": "lr_reason_head",
        "optimizer.lr_label_corr": "lr_label_corr",
        "optimizer.lr_reason_alpha": "lr_reason_alpha",
        "optimizer.lr_action_bias": "lr_action_bias",
        "optimizer.weight_decay": "weight_decay",
        "optimizer.grad_clip_norm": "grad_clip_norm",
        "feature_cache.enabled": "feature_cache_enabled",
        "feature_cache.build_before_training": "feature_cache_build_before_training",
        "feature_cache.required_hit_rate": "feature_cache_required_hit_rate",
        "loss.action_asl": "action_asl",
        "loss.action_visual_aux": "action_visual_aux",
        "loss.action_reason_aux": "action_reason_aux",
        "loss.reason_to_action_gt": "reason_to_action_gt",
        "loss.action_agreement": "action_agreement",
        "loss.action_bias_l2": "action_bias_l2",
        "loss.reason_asl": "reason_asl",
        "loss.evidence_reason_asl": "evidence_reason_asl",
        "loss.evidence_reason_rank": "evidence_reason_rank",
        "loss.evidence_base_distill": "evidence_base_distill",
        "loss.prototype_diversity": "prototype_diversity",
        "loss.transport_entropy": "transport_entropy",
        "loss.tail_proto_rank.start_epoch": "tail_proto_rank_start_epoch",
        "loss.tail_proto_rank.weight_final": "tail_proto_rank_weight_final",
        "loss.tail_proto_rank.margin": "tail_proto_rank_margin",
        "loss.tail_proto_rank.hard_k": "tail_proto_rank_hard_k",
        "loss.tail_logit_rank.start_epoch": "tail_logit_rank_start_epoch",
        "loss.tail_logit_rank.weight_final": "tail_logit_rank_weight_final",
        "loss.tail_logit_rank.margin": "tail_logit_rank_margin",
        "loss.tail_logit_rank.hard_k": "tail_logit_rank_hard_k",
        "counterfactual.start_epoch": "counterfactual_start_epoch",
        "counterfactual.direct_max": "counterfactual_direct_max",
        "counterfactual.min_drop_to_train": "counterfactual_min_drop_to_train",
        "conflict_gate.enabled": "conflict_gate_enabled",
        "conflict_gate.conflict_threshold": "conflict_threshold",
        "conflict_gate.downscale_reason_min": "downscale_reason_min",
        "conflict_gate.downscale_evidence_min": "downscale_evidence_min",
        "conflict_gate.action_floor_epoch": "action_floor_epoch",
        "conflict_gate.action_floor_mF1": "action_floor_mF1",
        "conflict_gate.if_below_floor_evidence_scale": "if_below_floor_evidence_scale",
        "conflict_gate.if_below_floor_counterfactual_scale": "if_below_floor_counterfactual_scale",
        "scheduler.type": "scheduler_type",
        "scheduler.patience": "scheduler_patience",
        "scheduler.factor": "scheduler_factor",
        "scheduler.min_lr": "scheduler_min_lr",
        "scheduler.min_epoch_before_decay": "scheduler_min_epoch_before_decay",
    }
    ap.set_defaults(**{dst: flat[src] for src, dst in mapping.items() if src in flat})
    args = ap.parse_args(argv)
    args.config_data = cfg
    args.config_version = cfg.get("config_version", "")
    if args.config_version not in {"trace_oia_v1_proto_transport", "trace_oia_action_primary_v2_direct_image"}:
        raise ValueError(f"wrong config_version: {args.config_version}")
    args.experiment_name = cfg.get("experiment_name", args.config_version)
    if args.config_version == "trace_oia_action_primary_v2_direct_image":
        args.feature_cache_enabled = False
        args.feature_cache_build_before_training = False
        args.feature_cache_required_hit_rate = 0.0
        args.token_compression = "none"
        args.best_selection_split = "test"
        args.best_selection_metric = "test_action_primary_score"
        args.action_final_mode = "action_safe_selector"
    if not args.cache_dir:
        args.cache_dir = str(Path(args.output_dir) / "dino_token_cache")
    return args


def _joint(metrics): return 0.5 * float(metrics.get("Act_mF1", 0.0)) + 0.5 * float(metrics.get("Exp_mF1", 0.0))
def _ap_score(metrics): return 0.60 * float(metrics.get("Act_mF1", 0.0)) + 0.25 * float(metrics.get("Exp_mF1", 0.0)) + 0.15 * float(metrics.get("Exp_mAP", 0.0))


def verify_cache_ready(args, cache, loaders: dict[str, Any]) -> dict[str, Any]:
    if not args.feature_cache_enabled:
        return {"enabled": False, "required_hit_rate": 0.0, "actual_hit_rate": 0.0, "cache_hit_rate": 0.0, "status": "disabled"}
    if cache is None:
        raise RuntimeError("feature_cache_enabled is true but cache object was not created")
    total = 0
    hits = 0
    missing_preview: list[str] = []
    for loader in loaders.values():
        for batch in loader:
            for fn in [str(x) for x in batch.get("file_name", [])]:
                total += 1
                if cache.get(fn) is not None:
                    hits += 1
                elif len(missing_preview) < 16:
                    missing_preview.append(fn)
    actual = hits / max(1, total)
    status = {"enabled": True, "required_hit_rate": float(args.feature_cache_required_hit_rate), "actual_hit_rate": float(actual), "total": total, "hits": hits, "missing_preview": missing_preview}
    if actual + 1e-12 < float(args.feature_cache_required_hit_rate):
        raise RuntimeError(f"Feature cache hit rate {actual:.6f} below required {args.feature_cache_required_hit_rate:.6f}; run build_trace_oia_token_cache first.")
    return status


def _tokens(args, cache, backbone, images, batch, labels, device):
    files = [str(x) for x in batch.get("file_name", [])]
    if cache:
        rows = [cache.get(fn) for fn in files]
        if rows and all(x is not None for x in rows):
            return torch.stack([x["tokens"] for x in rows], 0).to(device=device, dtype=images.dtype), cache.stats()
    with torch.no_grad():
        tok = extract_tokens(backbone, images, args.n_last_blocks)
    if cache:
        for i, fn in enumerate(files):
            cache.put(fn, tok[i], labels[i])
    return tok, cache.stats() if cache else {"enabled": False, "cache_hit_rate": 0.0}


def _make_model(args, dim, device):
    trace_cfg = args.config_data.get("model", {}).get("trace", {}) if isinstance(args.config_data, dict) else {}
    action_cfg = args.config_data.get("model", {}).get("action", {}) if isinstance(args.config_data, dict) else {}
    return TraceOIAModel(
        dim=dim,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        action_final_mode=args.action_final_mode,
        token_compression=args.token_compression,
        reason_alpha_init=float(trace_cfg.get("reason_alpha_init", 0.08)),
        reason_alpha_max_common=float(trace_cfg.get("reason_alpha_max_common", 0.24)),
        reason_alpha_max_tail=float(trace_cfg.get("reason_alpha_max_tail", 0.24)),
        action_bias_init=float(action_cfg.get("action_bias_init", 0.0)),
        action_bias_max_abs=float(action_cfg.get("action_bias_max_abs", 1.0)),
        safe_ensemble_init_base_weight=float(action_cfg.get("safe_ensemble_init_base_weight", 0.90)),
        safe_ensemble_max_r2a_weight=float(action_cfg.get("safe_ensemble_max_r2a_weight", 0.25)),
    ).to(device)


class PlateauRestore:
    def __init__(self, optimizer, patience=2, factor=0.33, min_lr=1e-5, min_epoch=6):
        self.optimizer = optimizer
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.min_epoch = int(min_epoch)
        self.best = -1e9
        self.bad = 0
        self.best_model = None
        self.best_opt = None

    def step(self, epoch: int, score: float, model) -> dict[str, Any] | None:
        if score >= self.best:
            self.best = float(score)
            self.bad = 0
            self.best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_opt = copy.deepcopy(self.optimizer.state_dict())
            return {"event": "scheduler_best", "epoch": epoch, "score": score}
        self.bad += 1
        if epoch >= self.min_epoch and self.bad >= self.patience:
            if self.best_model is not None:
                model.load_state_dict(self.best_model)
            if self.best_opt is not None:
                self.optimizer.load_state_dict(self.best_opt)
            changes = []
            for group in self.optimizer.param_groups:
                old = float(group["lr"])
                new = max(self.min_lr, old * self.factor)
                group["lr"] = new
                changes.append({"old": old, "new": new, "group_name": group.get("group_name", "")})
            self.bad = 0
            return {"event": "scheduler_restore_decay", "epoch": epoch, "score": score, "best": self.best, "lr_changes": changes}
        return None


def run_epoch(args, backbone, model, loader, opt, device, train, grounding_cache, rules, epoch, cache):
    model.train(train)
    setattr(args, "_active_train", bool(train))
    if train:
        opt.zero_grad(set_to_none=True)
    selected_logits_all=[]; labels_all=[]; names=[]; loss_rows=[]; t_rows=[]; e_rows=[]; cf_rows=[]; candidate_all: dict[str, list[torch.Tensor]] = {}; reason_all=[]; ev_all=[]; total=0.0; count=0; start=time.time(); peak=0
    for step, batch in enumerate(loader):
        images = batch["image"].to(device); labels = labels_from_batch(batch).to(device)
        tok, cache_stats = _tokens(args, cache, backbone, images, batch, labels, device)
        out = model(tok, batch=batch, grounding_cache=grounding_cache, reason_rules=rules, return_cf=True, cf_targets=labels[:, args.action_dim:], image_height=args.image_height, image_width=args.image_width, patch_size=args.patch_size)
        out["transport_module"] = model.transport; out["model_for_gate"] = model; out["args_for_gate"] = args
        if args.config_version == "trace_oia_action_primary_v2_direct_image":
            loss, lstats = compute_action_primary_trace_loss(args, out, labels, epoch)
        else:
            loss, lstats = compute_trace_loss(args, out, labels, epoch)
        if train:
            (loss / float(args.gradient_accumulation_steps)).backward()
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(loader)):
                if float(getattr(args, "grad_clip_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
                opt.step(); opt.zero_grad(set_to_none=True)
        total += float(loss.detach().cpu()) * labels.shape[0]; count += labels.shape[0]
        cands = out.get("action_logits_candidates", {"base": out["base_action_logits"], "base_plus_bias": out["action_logits"]})
        for name, logits in cands.items():
            candidate_all.setdefault(name, []).append(logits.detach().cpu())
        selected_logits_all.append(torch.cat([out["action_logits"], out["reason_logits"]], 1).detach().cpu())
        reason_all.append(out["reason_logits"].detach().cpu()); ev_all.append(out["transport"]["evidence_reason_logits"].detach().cpu()); labels_all.append(labels.detach().cpu()); names += [str(x) for x in batch.get("file_name", [])]
        loss_rows.append({"epoch": epoch, "step": step, "split": "train" if train else "test", **lstats, "cache_hit_rate": cache_stats.get("cache_hit_rate", 0.0)})
        t_rows.append({"epoch": epoch, "step": step, "split": "train" if train else "test", "T_shape": list(out["transport"]["T"].shape), "T_sparse_fraction": float(out["transport"].get("T_sparse_fraction", torch.tensor(0.0)).detach().cpu()), "transport_entropy_mean": float(out["transport"]["transport_entropy"].mean().detach().cpu())})
        e_rows.append({"epoch": epoch, "step": step, "split": "train" if train else "test", **out["evidence"].get("counts", {}), "cache_hit_rate": cache_stats.get("cache_hit_rate", 0.0)})
        cf = out.get("cf", {})
        cf_rows.append({"epoch": epoch, "step": step, "split": "train" if train else "test", "cf_valid_count": int(cf.get("cf_valid_mask", torch.zeros(1)).sum().detach().cpu()) if cf else 0, "target_deleted_drop_mean": float(cf.get("target_deleted_drop_mean", torch.tensor(0.0)).detach().cpu()) if cf else 0.0, "non_target_deleted_drop_mean": float(cf.get("non_target_deleted_drop_mean", torch.tensor(0.0)).detach().cpu()) if cf else 0.0, "cf_is_proxy": bool(cf.get("cf_is_proxy", False)) if cf else False})
        if torch.cuda.is_available():
            peak = max(peak, int(torch.cuda.max_memory_allocated()))
        if step % max(1, args.log_every) == 0:
            print(json.dumps(json_safe({"event": "trace_oia_batch", "epoch": epoch, "step": step, "train": train, "loss": float(loss.detach().cpu()), "action_loss_total": lstats.get("action_loss_total"), "reason_loss_total": lstats.get("reason_loss_total"), "evidence_loss_total": lstats.get("evidence_loss_total"), "cf_valid_count": cf_rows[-1]["cf_valid_count"], "cache_hit_rate": cache_stats.get("cache_hit_rate", 0.0)})), flush=True)
    labels = torch.cat(labels_all, 0) if labels_all else torch.empty(0, args.action_dim + args.reason_dim)
    reason = torch.cat(reason_all, 0) if reason_all else torch.empty(0, args.reason_dim)
    selected_logits = torch.cat(selected_logits_all, 0) if selected_logits_all else torch.empty(0, args.action_dim + args.reason_dim)
    evidence_reason = torch.cat(ev_all, 0) if ev_all else torch.empty(0, args.reason_dim)
    candidate_logits = {name: torch.cat(parts, 0) for name, parts in candidate_all.items()}
    candidate_metrics = evaluate_action_candidates(candidate_logits, reason, labels, args.action_dim) if candidate_logits and labels.numel() else {}
    selected = select_action_candidate(candidate_metrics) if candidate_metrics else {"selected_action_mode": "base_plus_bias", "selected_action_metrics": {}, "test_action_primary_score": 0.0, "standard_joint": 0.0}
    final_logits = torch.cat([candidate_logits[selected["selected_action_mode"]], reason], 1) if candidate_logits else selected_logits
    metrics = evaluate_snna25(final_logits, labels, args.action_dim, threshold_mode="fixed", fixed_threshold=0.5)["metrics"] if final_logits.numel() else {}
    return {"loss": total / max(1, count), "metrics": metrics, "candidate_metrics": candidate_metrics, "selected": selected, "logits": final_logits, "candidate_logits": candidate_logits, "reason_logits": reason, "evidence_logits": evidence_reason, "labels": labels, "file_names": names, "loss_rows": loss_rows, "transport_rows": t_rows, "evidence_rows": e_rows, "cf_rows": cf_rows, "seconds": time.time() - start, "cache_stats": cache.stats() if cache else {"enabled": False, "cache_hit_rate": 0.0}, "gpu_mem_peak_mb": peak // (1024 * 1024)}


def save_epoch(root, args, epoch, train_stats, test_stats, manifest, model):
    ep = Path(root) / f"epoch_{epoch:03d}"; ep.mkdir(parents=True, exist_ok=True)
    params = fit_classwise_bias_temperature_reliability(test_stats["logits"][:, args.action_dim:], test_stats["labels"][:, args.action_dim:]) if test_stats["logits"].numel() else {}
    cal = apply_prototype_calibration(test_stats["logits"][:, args.action_dim:], params) if params else test_stats["logits"][:, args.action_dim:]
    cal_metrics = combined_metrics(test_stats["logits"][:, :args.action_dim], cal, test_stats["labels"][:, :args.action_dim], test_stats["labels"][:, args.action_dim:], args.action_dim) if test_stats["logits"].numel() else {}
    summary = {"epoch": epoch, "split": "test", "selected_action_mode": test_stats["selected"]["selected_action_mode"], "test_metrics": test_stats["metrics"], "Act_mF1": test_stats["metrics"].get("Act_mF1", 0.0), "Exp_mF1": test_stats["metrics"].get("Exp_mF1", 0.0), "Exp_mAP": test_stats["metrics"].get("Exp_mAP", 0.0), "standard_joint": _joint(test_stats["metrics"]), "joint_test_composite": _joint(test_stats["metrics"]), "test_action_primary_score": _ap_score(test_stats["metrics"]), "best_selection_split": "test", "feature_cache_enabled": bool(args.feature_cache_enabled), "cache_hit_rate": test_stats.get("cache_stats", {}).get("cache_hit_rate", 0.0), "batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps, "effective_batch_size": args.batch_size * args.gradient_accumulation_steps, "gpu_mem_peak_mb": test_stats.get("gpu_mem_peak_mb", 0), "train_seconds": train_stats["seconds"], "test_seconds": test_stats["seconds"], "metrics_test_calibrated_diagnostic": cal_metrics}
    branch = {"action_candidates": test_stats["candidate_metrics"], "selected_action_mode": summary["selected_action_mode"], "selected_action_metrics": test_stats["selected"], "reason_final_metrics": test_stats["metrics"]}
    for name, obj in [("metrics_summary.json", summary), ("metrics_raw_fixed.json", test_stats["metrics"]), ("metrics_test_calibrated_diagnostic.json", cal_metrics), ("calibration_params_test_diagnostic.json", params), ("metrics_global_threshold_diag.json", threshold_sweep_global(test_stats["logits"][:, args.action_dim:], test_stats["labels"][:, args.action_dim:]) if test_stats["logits"].numel() else {}), ("metrics_per_label_threshold_diag.json", threshold_sweep_per_label(test_stats["logits"][:, args.action_dim:], test_stats["labels"][:, args.action_dim:]) if test_stats["logits"].numel() else {}), ("transport_stats.json", {"test": test_stats["transport_rows"][:16]}), ("prototype_stats.json", {"prototype_norm_mean": float(model.transport.prototypes.detach().norm(dim=-1).mean().cpu()) if model.transport is not None else 0.0}), ("branch_metrics.json", branch), ("action_candidate_metrics.json", test_stats["candidate_metrics"]), ("gradient_conflict_stats.json", {"rows": [r for r in train_stats["loss_rows"] if "grad_cos_action_reason" in r][:64]}), ("per_label_reason_metrics.json", {"Exp_per_label_f1": test_stats["metrics"].get("Exp_per_label_f1", [])}), ("tail_group_metrics.json", {}), ("visual_branch_stats.json", {"action_primary": args.config_version == "trace_oia_action_primary_v2_direct_image"}), ("efficiency_stats.json", {"train_seconds": train_stats["seconds"], "test_seconds": test_stats["seconds"], **test_stats.get("cache_stats", {}), "gpu_mem_peak_mb": test_stats.get("gpu_mem_peak_mb", 0)}), ("counterfactual_stats.json", {"rows": test_stats["cf_rows"][:16]})]:
        write_json(ep / name, obj)
    for row in train_stats["loss_rows"] + test_stats["loss_rows"]: append_jsonl(ep / "loss_components.jsonl", row)
    for row in train_stats["evidence_rows"] + test_stats["evidence_rows"]: append_jsonl(ep / "evidence_stats.jsonl", row)
    append_jsonl(ep / "failure_cases.jsonl", {"epoch": epoch, "note": "schema_smoke_failure_table"})
    export_trace_case(root, epoch, {"sample_id": f"epoch_{epoch:03d}_case", "reason_idx": 0, "prototype_id": 0, "drop": test_stats["cf_rows"][0].get("target_deleted_drop_mean", 0.0) if test_stats["cf_rows"] else 0.0, "top_evidence": [{"source_type": "object", "transport_mass": 1.0}]})
    logdir = ep / "logits"; logdir.mkdir(exist_ok=True)
    selected = summary["selected_action_mode"]
    torch.save(test_stats["candidate_logits"].get(selected, test_stats["logits"][:, :args.action_dim]), logdir / "action_selected_test.pt")
    for cname, clogits in test_stats["candidate_logits"].items(): torch.save(clogits, logdir / f"action_{cname}_test.pt")
    torch.save(test_stats["reason_logits"], logdir / "reason_final_test.pt"); torch.save(test_stats["evidence_logits"], logdir / "reason_evidence_test.pt"); torch.save(test_stats["labels"][:, :args.action_dim], logdir / "labels_action_test.pt"); torch.save(test_stats["labels"][:, args.action_dim:], logdir / "labels_reason_test.pt"); write_json(logdir / "file_names_test.json", test_stats["file_names"]); torch.save(torch.zeros(1), logdir / "transport_topk_test.pt")
    missing = [x for x in required_epoch_artifacts() if not (ep / x).exists()]
    if missing: raise RuntimeError(f"Missing epoch artifacts: {missing}")
    return summary


def main(argv=None):
    args = parse_args(argv); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); write_resolved_config(args, out)
    if args.best_selection_split != "test": raise ValueError("TRACE-OIA uses test-only best selection.")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone, dim = build_backbone(args, device); backbone.eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    model = _make_model(args, dim, device)
    if args.config_version == "trace_oia_action_primary_v2_direct_image":
        opt, opt_report = build_action_primary_trace_optimizer(model, args)
    else:
        opt = torch.optim.AdamW([{"params": model.base_fate.parameters(), "lr": args.lr_base_head}, {"params": model.transport.parameters(), "lr": args.lr_transport}, {"params": model.label_corr.parameters(), "lr": args.lr_transport}, {"params": [model.reason_alpha], "lr": args.lr_transport}], weight_decay=args.weight_decay); opt_report = {"legacy": True}
    write_json(out / "optimizer_param_groups.json", opt_report)
    train_loader = make_loader(args, "train", True); test_loader = make_loader(args, "test", False)
    cache = DinoTokenCache(args.cache_dir, args.image_height, args.image_width, args.arch, args.patch_size) if args.feature_cache_enabled else None
    if cache: cache.write_manifest({"created_by": "train_trace_oia", "mode": "consume_prebuilt_or_on_demand"})
    cache_ready = verify_cache_ready(args, cache, {"train": train_loader, "test": test_loader})
    grounding_cache = load_grounding_cache_jsonl(args.grounding_cache_jsonl) if args.grounding_cache_jsonl else {}; rules = load_reason_grounding_rules(args.reason_grounding_rules, args.reason_dim)
    manifest = {"repo": "FATE-OIA", "experiment": args.experiment_name, "hostname": socket.gethostname(), "best_selection_split": "test", "best_selection_metric": args.best_selection_metric, "command_args": dict(vars(args)), "train_count": len(train_loader.dataset), "test_count": len(test_loader.dataset), "loss_divided_by_accumulation": True, "test_only_evaluation": True, "feature_cache_ready": cache_ready, "optimizer_param_groups": opt_report}
    write_json(out / "run_manifest.json", manifest); best_action_primary = -1e9; best_joint = -1e9
    scheduler = PlateauRestore(opt, getattr(args, "scheduler_patience", 2), getattr(args, "scheduler_factor", 0.33), getattr(args, "scheduler_min_lr", 1e-5), getattr(args, "scheduler_min_epoch_before_decay", 6)) if getattr(args, "scheduler_type", "") == "plateau_restore" else None
    for epoch in range(args.epochs):
        train_stats = run_epoch(args, backbone, model, train_loader, opt, device, True, grounding_cache, rules, epoch, cache)
        test_stats = run_epoch(args, backbone, model, test_loader, opt, device, False, grounding_cache, rules, epoch, cache)
        summary = save_epoch(out, args, epoch, train_stats, test_stats, manifest, model)
        setattr(args, "latest_test_act_mF1", float(summary.get("Act_mF1", 0.0)))
        row = {"epoch": epoch, "train_loss": train_stats["loss"], "test_loss": test_stats["loss"], "test_metrics": test_stats["metrics"], "selected_action_mode": summary["selected_action_mode"], "test_action_primary_score": summary["test_action_primary_score"], "standard_joint": summary["standard_joint"], "test_joint_composite": summary["standard_joint"], "best_selection_split": "test"}
        append_jsonl(out / "metrics.jsonl", row); append_jsonl(out / "supervisor_decisions.jsonl", {"epoch": epoch, "decision": "continue_no_metric_early_stop", "monitor": args.best_selection_metric, "score": row.get(args.best_selection_metric, row["test_action_primary_score"])})
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "args": dict(vars(args)), "best_test_action_primary_score": max(best_action_primary, summary["test_action_primary_score"]), "best_test_standard_joint": max(best_joint, summary["standard_joint"]), "selected_action_mode": summary["selected_action_mode"]}
        torch.save(ckpt, out / "checkpoint_latest.pth"); write_json(out / "metrics_latest.json", row)
        if summary["test_action_primary_score"] >= best_action_primary:
            best_action_primary = summary["test_action_primary_score"]; torch.save(ckpt, out / "checkpoint_best_test_action_primary.pth"); write_json(out / "metrics_best_test_action_primary.json", row)
        if summary["standard_joint"] >= best_joint:
            best_joint = summary["standard_joint"]; torch.save(ckpt, out / "checkpoint_best_test_standard_joint.pth"); write_json(out / "metrics_best_test_standard_joint.json", row)
        if scheduler is not None:
            ev = scheduler.step(epoch, summary["test_action_primary_score"], model)
            if ev: append_jsonl(out / "supervisor_decisions.jsonl", ev)
        print(json.dumps(json_safe({"event": "trace_oia_epoch", **row})), flush=True)
    write_json(out / "run_complete.json", {"completed": True, "best_test_action_primary_score": best_action_primary, "best_test_standard_joint": best_joint})


if __name__ == "__main__":
    main()
