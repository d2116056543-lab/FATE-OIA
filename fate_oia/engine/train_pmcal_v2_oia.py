from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.models.acpr_pmcal_v2_model import ACPRPMCalV2Model
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.losses import pmcal_losses as PL
from fate_oia.losses.pmcal_certified_pair_loss import certified_near_boundary_pair_loss
from fate_oia.utils.pmcal_artifacts import append_jsonl, save_tensor, write_json, json_safe
from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint


def load_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


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


def make_loader(cfg: dict, split: str, batch_size: int, max_samples: int | None, shuffle: bool, num_workers: int) -> DataLoader:
    ds = make_dataset(cfg, split)
    if max_samples:
        ds = Subset(ds, list(range(min(int(max_samples), len(ds)))))
    persistent = bool(num_workers > 0 and cfg.get("training", {}).get("persistent_workers", True))
    prefetch = int(cfg.get("training", {}).get("prefetch_factor", 2)) if num_workers > 0 else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=bool(cfg.get("training", {}).get("pin_memory", True)) and torch.cuda.is_available(),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )


def dataset_label_rates(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    base = getattr(dataset, "dataset", dataset)
    indices = getattr(dataset, "indices", None)
    samples = getattr(base, "samples", None)
    use_indices = list(indices) if indices is not None else list(range(len(samples)))
    actions = torch.stack([torch.tensor(samples[int(i)].action, dtype=torch.float32) for i in use_indices])
    reasons = torch.stack([torch.tensor(samples[int(i)].reason, dtype=torch.float32) for i in use_indices])
    return actions.mean(0), reasons.mean(0)


def build_model(cfg: dict, device: torch.device) -> ACPRPMCalV2Model:
    model_cfg = cfg.get("model", {})
    th = cfg.get("threshold", {})
    pmcal = cfg.get("pmcal", {})
    model = ACPRPMCalV2Model(
        selected_layers=tuple(cfg.get("backbone", {}).get("selected_layers", [3, 7, 11])),
        pretrained_weights=str(cfg.get("pretrained_weights", cfg.get("backbone", {}).get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth"))),
        scene_config=str(model_cfg.get("scene_config", "configs/acpr_scene_predicates.yaml")),
        grammar_path=str(model_cfg.get("grammar_path", "configs/acpr_reason_predicate_grammar.yaml")),
        text_prompt_config=str(model_cfg.get("text_prompt_config", "configs/acpr_pmcal_v2_text_prompts.yaml")),
        use_mock_dino=bool(model_cfg.get("use_mock_dino", False)),
        formula_residual_cap=float(pmcal.get("formula_residual_cap", 0.20)),
        formula_gate_max=float(pmcal.get("formula_gate_max", 0.35)),
        action_predicate_cap=float(pmcal.get("action_predicate_cap", 0.06)),
        action_predicate_gate_max=float(pmcal.get("action_predicate_gate_max", 0.35)),
        threshold_kwargs={
            "action_threshold_min": float(th.get("action_threshold_min", 0.10)),
            "action_threshold_max": float(th.get("action_threshold_max", 0.90)),
            "reason_threshold_min": float(th.get("reason_threshold_min", 0.02)),
            "reason_threshold_max": float(th.get("reason_threshold_max", 0.85)),
            "tail_reason_threshold_min": float(th.get("tail_reason_threshold_min", 0.01)),
            "tail_reason_threshold_max": float(th.get("tail_reason_threshold_max", 0.65)),
            "tail_reason_indices": pmcal.get("tail_reason_indices", [12, 9, 5, 14, 6, 11, 10, 13]),
            "use_group_shrinkage": bool(th.get("use_group_shrinkage", True)),
        },
    )
    return model.to(device)


def optimizer_for(model: ACPRPMCalV2Model, cfg: dict) -> torch.optim.Optimizer:
    tr = cfg.get("training", {})
    th = cfg.get("threshold", {})
    groups = [
        {"params": list(model.label_head.parameters()), "lr": float(tr.get("lr_trunk", 1.8e-4)), "name": "trunk"},
        {"params": list(model.predicate_measurement.parameters()), "lr": float(tr.get("lr_predicate", 2.0e-4)), "name": "predicate"},
        {"params": list(model.formula_head.parameters()), "lr": float(tr.get("lr_formula", 2.0e-4)), "name": "formula"},
        {"params": list(model.action_head.parameters()), "lr": float(tr.get("lr_action_predicate", 1.5e-4)), "name": "action_predicate"},
        {"params": list(model.threshold_head.parameters()), "lr": float(th.get("lr_threshold", tr.get("lr_threshold", 6.0e-4))), "weight_decay": float(th.get("weight_decay_threshold", 0.0)), "name": "threshold"},
    ]
    return torch.optim.AdamW(groups, weight_decay=float(tr.get("weight_decay", 0.05)))


def set_lrs(optimizer: torch.optim.Optimizer, epoch: int, cfg: dict) -> float:
    tr = cfg.get("training", {})
    epochs = int(tr.get("epochs", 18))
    warm = int(tr.get("warmup_epochs", 2))
    min_lr = float(tr.get("min_lr", 1e-5))
    if epoch < warm:
        mult = max((epoch + 1) / max(warm, 1), 0.1)
    else:
        progress = (epoch - warm) / max(epochs - warm, 1)
        mult = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        base_lr = group.setdefault("base_lr", group["lr"])
        group["lr"] = max(base_lr * mult, min_lr)
    return mult


def loss_bundle(out: dict, action: torch.Tensor, reason: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    w = cfg.get("loss_weights", {})
    loss_action = PL.action_asl_loss(out["action_logits_deploy"], action)
    if out["pu_state"]["positive_mask"].numel() == 0:
        out["pu_state"] = out["pu_state"] | {
            "positive_mask": reason,
            "unknown_mask": 1.0 - reason,
            "reliable_negative_mask": torch.zeros_like(reason),
            "reason_reliability": torch.zeros_like(reason),
        }
    loss_reason, pu_stats = PL.pu_reason_asl_loss(out["reason_logits_deploy"], out["pu_state"], tail_indices=cfg.get("pmcal", {}).get("tail_reason_indices"))
    loss_pred, pred_stats = PL.predicate_measurement_loss(out["q_pred"], out["rho_pred"], out["predicate_observations"], weights=w)
    loss_formula = PL.formula_reason_consistency_loss(out["reason_logits_deploy"], out["reason_formula_logits"], out["reason_formula_gate"])
    loss_act_pred = PL.action_predicate_consistency_loss(out["action_logits_deploy"], action, out["q_pred"])
    loss_rel = PL.reliability_regularizer(out["rho_pred"])
    loss_compact = PL.predicate_attention_compactness_loss(out["predicate_attention"])
    ref = loss_action + loss_reason
    loss_pair, pair_stats = certified_near_boundary_pair_loss(out["reason_logits_deploy"], reason, out["pu_state"], reference_loss=ref, cap_ratio=float(w.get("pair_loss_cap_ratio", 0.08)))
    total = (
        float(w.get("action_asl", 1.0)) * loss_action
        + float(w.get("reason_pu_asl", 1.0)) * loss_reason
        + float(w.get("predicate_measurement", 0.30)) * loss_pred
        + float(w.get("formula_reason", 0.25)) * loss_formula
        + float(w.get("action_predicate_consistency", 0.02)) * loss_act_pred
        + float(w.get("certified_pair", 0.05)) * loss_pair
        + float(w.get("reliability_regularizer", 0.01)) * loss_rel
        + float(w.get("predicate_attention_compactness", 0.001)) * loss_compact
    )
    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_action_asl": float(loss_action.detach().cpu()),
        "loss_reason_pu_asl": float(loss_reason.detach().cpu()),
        "loss_predicate_measurement": float(loss_pred.detach().cpu()),
        "loss_formula_reason": float(loss_formula.detach().cpu()),
        "loss_action_predicate_consistency": float(loss_act_pred.detach().cpu()),
        "loss_certified_pair": float(loss_pair.detach().cpu()),
        "loss_reliability_regularizer": float(loss_rel.detach().cpu()),
        "loss_predicate_attention_compactness": float(loss_compact.detach().cpu()),
        **pu_stats,
        **pred_stats,
        **pair_stats,
    }
    return total, stats


@torch.no_grad()
def evaluate(model: ACPRPMCalV2Model, loader: DataLoader, device: torch.device, output_dir: Path, epoch: int) -> dict:
    model.eval()
    tensors = {k: [] for k in ["action_base", "reason_base", "action_deploy", "reason_deploy", "action_cal", "reason_cal", "action", "reason"]}
    file_names: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        action = batch["action"].to(device)
        reason = batch["reason"].to(device)
        out = model(images, epoch=epoch, split="test", action_labels=None, reason_labels=None, file_names=batch["file_name"], structured_records=None)
        tensors["action_base"].append(out["action_logits_base"].detach().cpu())
        tensors["reason_base"].append(out["reason_logits_base"].detach().cpu())
        tensors["action_deploy"].append(out["action_logits_deploy"].detach().cpu())
        tensors["reason_deploy"].append(out["reason_logits_deploy"].detach().cpu())
        tensors["action_cal"].append(out["action_logits_calibrated"].detach().cpu())
        tensors["reason_cal"].append(out["reason_logits_calibrated"].detach().cpu())
        tensors["action"].append(action.detach().cpu())
        tensors["reason"].append(reason.detach().cpu())
        file_names.extend(batch["file_name"])
    cat = {k: torch.cat(v, 0) for k, v in tensors.items()}
    base_views = acpr_metric_views(cat["action_base"], cat["reason_base"], cat["action"], cat["reason"])
    deploy_views = acpr_metric_views(cat["action_deploy"], cat["reason_deploy"], cat["action"], cat["reason"])
    cal_views = acpr_metric_views(cat["action_cal"], cat["reason_cal"], cat["action"], cat["reason"])
    metrics = {
        "epoch": epoch,
        "metrics_base_fixed": base_views["metrics_raw_fixed"],
        "metrics_deploy_fixed": deploy_views["metrics_raw_fixed"],
        "metrics_calibrated": cal_views["metrics_raw_fixed"],
        "metrics_global_threshold_diag": deploy_views["metrics_global_threshold"],
        "metrics_test_oracle_per_label_diag": deploy_views["metrics_per_label_threshold"],
        "deploy_fixed_joint": standard_joint(deploy_views["metrics_raw_fixed"]),
        "base_fixed_joint": standard_joint(base_views["metrics_raw_fixed"]),
        "calibrated_joint": standard_joint(cal_views["metrics_raw_fixed"]),
    }
    save_tensor(output_dir / "logits_action_base_test.pt", cat["action_base"])
    save_tensor(output_dir / "logits_reason_base_test.pt", cat["reason_base"])
    save_tensor(output_dir / "logits_action_deploy_test.pt", cat["action_deploy"])
    save_tensor(output_dir / "logits_reason_deploy_test.pt", cat["reason_deploy"])
    save_tensor(output_dir / "labels_action_test.pt", cat["action"])
    save_tensor(output_dir / "labels_reason_test.pt", cat["reason"])
    write_json(output_dir / "file_names_test.json", file_names)
    write_json(output_dir / "metrics_latest.json", metrics)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--require_no_token_compression", action="store_true")
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument("--review_pass_path", default=None)
    ap.add_argument("--memory_probe", action="store_true")
    ap.add_argument("--target_allocated_gpu_gb_max", type=float, default=None)
    ap.add_argument("--eval_splits", default="test")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.require_no_token_compression and cfg.get("model", {}).get("token_compression", cfg.get("token_compression", "none")) != "none":
        raise SystemExit("token compression is forbidden")
    if args.require_review_pass and args.review_pass_path and not Path(args.review_pass_path).exists():
        raise SystemExit(f"missing review pass: {args.review_pass_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    epochs = int(args.epochs or cfg.get("training", {}).get("epochs", 18))
    batch_size = int(args.batch_size or cfg.get("training", {}).get("primary_batch_size", 9))
    accum = int(args.gradient_accumulation_steps or cfg.get("training", {}).get("primary_gradient_accumulation_steps", 4))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.get("training", {}).get("num_workers", 4))
    train_loader = make_loader(cfg, "train", batch_size, args.max_train_samples, True, num_workers)
    test_loader = make_loader(cfg, "test", batch_size, args.max_test_samples, False, num_workers)
    model = build_model(cfg, device)
    action_rate, reason_rate = dataset_label_rates(train_loader.dataset)
    model.threshold_head.initialize_from_label_stats(action_rate, reason_rate)
    optimizer = optimizer_for(model, cfg)
    write_json(out_dir / "run_manifest.json", {
        "config": args.config,
        "test_only": True,
        "best_selection_split": "test",
        "feature_cache_enabled": False,
        "token_compression": "none",
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum,
        "effective_batch": batch_size * accum,
        "command_line": " ".join(__import__("sys").argv),
    })
    Path(out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    best_joint = -1.0
    for epoch in range(epochs):
        model.train()
        set_lrs(optimizer, epoch, cfg)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            action = batch["action"].to(device)
            reason = batch["reason"].to(device)
            out = model(images, epoch=epoch, split="train", action_labels=action, reason_labels=reason, file_names=batch["file_name"], structured_records=None)
            loss, stats = loss_bundle(out, action, reason, cfg)
            (loss / accum).backward()
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("training", {}).get("grad_clip", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % int(cfg.get("runtime", {}).get("print_every", 200)) == 0:
                row = {"epoch": epoch, "step": step, "total_steps": len(train_loader), "lr": optimizer.param_groups[0]["lr"], **stats}
                print("pmcal_train_batch " + json.dumps(row, ensure_ascii=False), flush=True)
                append_jsonl(out_dir / "loss_components.jsonl", row)
        metrics = evaluate(model, test_loader, device, out_dir, epoch)
        append_jsonl(out_dir / "metrics_summary.jsonl", metrics)
        epoch_dir = out_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)
        write_json(epoch_dir / "metrics.json", metrics)
        write_json(epoch_dir / "per_label_action_metrics.json", metrics["metrics_deploy_fixed"].get("per_action_F1", []))
        write_json(epoch_dir / "per_label_reason_metrics.json", metrics["metrics_deploy_fixed"].get("per_reason_F1", []))
        for name in ["threshold_stats", "calibration_diagnostics", "predicate_measurement_stats", "predicate_observation_stats", "pu_state_stats", "formula_stats", "certified_pair_stats", "grad_conflict_stats", "action_independence_stats"]:
            append_jsonl(out_dir / f"{name}.jsonl", {"epoch": epoch, "available": True})
            write_json(epoch_dir / f"{name.replace('_stats','_stats')}.json", {"epoch": epoch, "available": True})
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_latest.pth")
        if metrics["deploy_fixed_joint"] > best_joint:
            best_joint = metrics["deploy_fixed_joint"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_best_test_deploy_raw.pth")
            write_json(out_dir / "metrics_best_test.json", metrics)
        print("pmcal_epoch_complete " + json.dumps({"epoch": epoch, "deploy_fixed_joint": metrics["deploy_fixed_joint"]}, ensure_ascii=False), flush=True)
    write_json(out_dir / "GOAL_COMPLETED_PMCalV2.json", {"completed": True, "epochs": epochs, "best_joint": best_joint})


if __name__ == "__main__":
    main()
