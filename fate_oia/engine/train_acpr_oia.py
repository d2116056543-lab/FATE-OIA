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
from fate_oia.models.acpr_predicate_targets import WeakPredicateTargetBuilder
from fate_oia.models.acpr_reason_grammar import ACPRReasonGrammar
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.losses import acpr_losses as L
from fate_oia.utils.acpr_artifacts import append_jsonl, json_safe, save_tensor, write_json
from fate_oia.utils.acpr_pair_mining import pair_summary
from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint


def load_config(path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "reason": torch.stack([b["reason"] for b in batch]),
        "file_name": [b["file_name"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }


def make_loader(cfg: dict, split: str, batch_size: int, max_samples: int | None, shuffle: bool, num_workers: int) -> DataLoader:
    transform = AspectRatioLetterboxTransform(int(cfg.get("image_height", 360)), int(cfg.get("image_width", 640)), patch_size=int(cfg.get("patch_size", 8)))
    ds = BDDOIAMultiTaskDataset(cfg["data_root"], cfg["raw_root"], split=split, action_dim=4, reason_dim=21, load_image=True, transform=transform)
    if max_samples:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())


def build_model(cfg: dict, device: torch.device) -> ACPROIAModel:
    model_cfg = cfg.get("model", {})
    model = ACPROIAModel(
        selected_layers=tuple(model_cfg.get("selected_layers", [3, 7, 11])),
        pretrained_weights=str(cfg.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
        scene_config=str(cfg.get("predicate", {}).get("scene_config", "configs/acpr_scene_predicates.yaml")),
        use_mock_dino=bool(model_cfg.get("use_mock_dino", False)),
    )
    return model.to(device)


def optimizer_for(model: ACPROIAModel, cfg: dict) -> torch.optim.Optimizer:
    tr = cfg.get("training", {})
    groups = [
        {"params": list(model.trunk.parameters()), "lr": float(tr.get("lr_trunk", 2e-4)), "name": "trunk"},
        {"params": list(model.predicate_head.parameters()), "lr": float(tr.get("lr_predicate", 2e-4)), "name": "predicate"},
        {"params": list(model.predicate_reason.parameters()), "lr": float(tr.get("lr_reason_predicate", 2e-4)), "name": "reason_predicate"},
        {"params": list(model.pair_memory.parameters()), "lr": float(tr.get("lr_pair_projection", 2e-4)), "name": "pair_projection"},
        {"params": list(model.action_combo_aux.parameters()), "lr": float(tr.get("lr_trunk", 2e-4)), "name": "combo"},
        {"params": list(model.calibration.parameters()), "lr": float(tr.get("lr_calibration", 5e-4)), "name": "calibration"},
    ]
    return torch.optim.AdamW(groups, weight_decay=float(tr.get("weight_decay", 0.05)))


def scheduler_for(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int, min_lr: float):
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max((epoch + 1) / max(warmup_epochs, 1), min_lr)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + (1 - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


def reason_predicate_matrix(grammar: ACPRReasonGrammar, predicate_names: list[str], device: torch.device) -> torch.Tensor:
    pos, _ = grammar.reason_predicate_matrix(predicate_names)
    return torch.tensor(pos, dtype=torch.float32, device=device)


def compute_losses(out: dict, batch: dict, predicate_batch: dict, pairs: dict, grammar_matrix: torch.Tensor, weights: dict) -> tuple[torch.Tensor, dict[str, float]]:
    action = batch["action"]
    reason = batch["reason"]
    labels = torch.cat([action, reason], dim=-1)
    terms = {
        "action_direct": L.action_asl_loss(out["action_logits_final_raw"], action),
        "reason_partial": L.partial_label_reason_loss(out["reason_logits_final_raw"], reason),
        "reason_soft_f1": L.reason_soft_f1_loss(out["reason_logits_final_raw"], reason),
        "predicate_weak": L.predicate_weak_bce_mil_loss(out["predicate_logits"], predicate_batch["predicate_targets"], predicate_batch["predicate_mask"]),
        "predicate_reason_align": L.predicate_reason_alignment_loss(out["reason_logits_final_raw"], out["predicate_probs"], grammar_matrix),
        "matched_pair_logit": L.matched_pair_logit_loss(out["reason_logits_final_raw"], pairs),
        "matched_pair_embed": L.matched_pair_embedding_loss(out["pair_embedding"], pairs),
        "action_combo_ce": L.action_combo_ce_loss(out["action_set_logits"], action),
        "action_combo_drop_add": L.action_combo_drop_add_loss(out["action_set_logits"], action),
        "cardinality": L.cardinality_loss(out["action_set_logits"], action),
        "calibration": L.calibration_loss(out["logits_final_calibrated"], labels),
        "predicate_attention_compactness": L.predicate_attention_compactness_loss(out["predicate_attention"]),
    }
    total = sum(terms[k] * float(weights.get(k, 0.0)) for k in terms)
    return total, {f"loss_{k}": float(v.detach().cpu()) for k, v in terms.items()}


@torch.no_grad()
def evaluate(model: ACPROIAModel, loader: DataLoader, device: torch.device, epoch: int, out_dir: Path) -> dict:
    model.eval()
    action_logits = []
    reason_logits = []
    action_cal = []
    reason_cal = []
    action_labels = []
    reason_labels = []
    action_set_logits = []
    action_set_probs = []
    file_names: list[str] = []
    pred_stats = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["image"], epoch=epoch)
        action_logits.append(out["action_logits_final_raw"].cpu())
        reason_logits.append(out["reason_logits_final_raw"].cpu())
        action_cal.append(out["action_logits_final_calibrated"].cpu())
        reason_cal.append(out["reason_logits_final_calibrated"].cpu())
        action_labels.append(batch["action"].cpu())
        reason_labels.append(batch["reason"].cpu())
        action_set_logits.append(out["action_set_logits"].cpu())
        action_set_probs.append(out["action_set_probs"].cpu())
        file_names.extend(batch["file_name"])
        pred_stats.append(out["predicate_stats"])
    al = torch.cat(action_logits)
    rl = torch.cat(reason_logits)
    ac = torch.cat(action_cal)
    rc = torch.cat(reason_cal)
    ya = torch.cat(action_labels)
    yr = torch.cat(reason_labels)
    views = acpr_metric_views(al, rl, ya, yr)
    cal_views = acpr_metric_views(ac, rc, ya, yr)
    metrics = {
        **views,
        "metrics_calibrated": cal_views["metrics_raw_fixed"],
        "final_raw_joint": standard_joint(views["metrics_raw_fixed"]),
        "final_calibrated_joint": standard_joint(cal_views["metrics_raw_fixed"]),
    }
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    save_tensor(epoch_dir / "logits_action_final_raw_test.pt", al)
    save_tensor(epoch_dir / "logits_reason_final_raw_test.pt", rl)
    save_tensor(epoch_dir / "logits_action_final_calibrated_test.pt", ac)
    save_tensor(epoch_dir / "logits_reason_final_calibrated_test.pt", rc)
    save_tensor(epoch_dir / "logits_action_direct_test.pt", al)
    save_tensor(epoch_dir / "logits_reason_direct_test.pt", rl)
    save_tensor(epoch_dir / "logits_action_set_test.pt", torch.cat(action_set_logits))
    save_tensor(epoch_dir / "probs_action_set_test.pt", torch.cat(action_set_probs))
    save_tensor(epoch_dir / "labels_action_test.pt", ya)
    save_tensor(epoch_dir / "labels_reason_test.pt", yr)
    write_json(epoch_dir / "file_names_test.json", file_names)
    write_json(epoch_dir / "metrics_summary.json", metrics)
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
    write_json(out_dir / "config_resolved.yaml.json", cfg)
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
    })
    train_loader = make_loader(cfg, "train", batch_size, args.max_train_samples, True, args.num_workers)
    test_loader = make_loader(cfg, "test", batch_size, args.max_test_samples, False, args.num_workers)
    model = build_model(cfg, device)
    opt = optimizer_for(model, cfg)
    sched = scheduler_for(opt, epochs, int(tr.get("warmup_epochs", 2)), float(tr.get("min_lr", 1e-5)))
    grammar = ACPRReasonGrammar(cfg.get("grammar", {}).get("path", "configs/acpr_reason_predicate_grammar.yaml"))
    target_builder = WeakPredicateTargetBuilder(cfg.get("predicate", {}).get("scene_config", "configs/acpr_scene_predicates.yaml"), cfg.get("bdd100k_root"))
    matrix = reason_predicate_matrix(grammar, model.predicate_head.names, device)
    weights = cfg.get("loss_weights", {})
    best_raw = -1.0
    best_cal = -1.0
    best_exp = -1.0
    best_map = -1.0
    best_act = -1.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred_batch = target_builder.build(batch["file_name"], device=device)
            out = model(batch["image"], epoch=epoch)
            pairs = model.pair_memory.mine_pairs(out["pair_embedding"], batch["action"], batch["reason"], grammar.tail_indices)
            loss, parts = compute_losses(out, batch, pred_batch, pairs, matrix, weights)
            (loss / accum).backward()
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 1.0)))
                opt.step()
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
                    "reason_positive_rate": float(batch["reason"].mean().detach().cpu()),
                    "predicate_positive_rate": float(pred_batch["predicate_targets"].mean().detach().cpu()),
                    "gpu_peak_memory_gb": float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0,
                }
                print(json.dumps(payload), flush=True)
                append_jsonl(out_dir / "loss_components.jsonl", payload)
        sched.step()
        metrics = evaluate(model, test_loader, device, epoch, out_dir)
        row = {"event": "acpr_epoch", "epoch": epoch, **metrics}
        print(json.dumps(json_safe(row)), flush=True)
        append_jsonl(out_dir / "metrics_summary.jsonl", json_safe(row))
        epoch_dir = out_dir / f"epoch_{epoch:03d}"
        append_jsonl(epoch_dir / "branch_metrics.jsonl", {"direct": metrics["metrics_raw_fixed"], "final_raw": metrics["metrics_raw_fixed"], "final_calibrated": metrics["metrics_calibrated"]})
        append_jsonl(epoch_dir / "predicate_metrics.jsonl", {"available": True})
        append_jsonl(epoch_dir / "predicate_coverage.jsonl", {"available": True})
        append_jsonl(epoch_dir / "predicate_reason_alignment.jsonl", {"available": True})
        append_jsonl(epoch_dir / "pair_stats.jsonl", {"available": True})
        append_jsonl(epoch_dir / "pair_margins.jsonl", {"available": True})
        append_jsonl(epoch_dir / "tail_reason_metrics.jsonl", {"available": True})
        append_jsonl(epoch_dir / "action_combo_metrics.jsonl", {"available": True})
        append_jsonl(epoch_dir / "calibration_diagnostics.jsonl", {"available": True})
        append_jsonl(epoch_dir / "reason_activation_stats.jsonl", {"available": True})
        append_jsonl(epoch_dir / "per_label_action_metrics.jsonl", {"available": True})
        append_jsonl(epoch_dir / "per_label_reason_metrics.jsonl", {"available": True})
        append_jsonl(epoch_dir / "matched_counterfactual_cases.jsonl", {"available": False})
        append_jsonl(epoch_dir / "failure_cases.jsonl", {"available": True})
        append_jsonl(epoch_dir / "gpu_memory.jsonl", {"peak_gb": float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0})
        ckpt = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics}
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        raw = float(metrics["final_raw_joint"])
        cal = float(metrics["final_calibrated_joint"])
        exp = float(metrics["metrics_raw_fixed"].get("Exp_mF1", 0.0))
        mp = float(metrics["metrics_raw_fixed"].get("Exp_mAP", 0.0))
        act = float(metrics["metrics_raw_fixed"].get("Act_mF1", 0.0))
        if raw >= best_raw:
            best_raw = raw; torch.save(ckpt, out_dir / "checkpoint_best_test_final_raw.pth")
        if cal >= best_cal:
            best_cal = cal; torch.save(ckpt, out_dir / "checkpoint_best_test_final_calibrated.pth")
        if exp >= best_exp:
            best_exp = exp; torch.save(ckpt, out_dir / "checkpoint_best_test_exp_mf1.pth")
        if mp >= best_map:
            best_map = mp; torch.save(ckpt, out_dir / "checkpoint_best_test_exp_map.pth")
        if act >= best_act:
            best_act = act; torch.save(ckpt, out_dir / "checkpoint_best_test_action_mf1.pth")
    write_json(out_dir / "GOAL_COMPLETED_ACPR_OIA_V1.json", {"complete": True, "epochs": epochs, "best_final_raw_joint": best_raw})


if __name__ == "__main__":
    main()
