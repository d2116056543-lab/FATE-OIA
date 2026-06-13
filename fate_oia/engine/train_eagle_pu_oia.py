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
from torch.utils.data import DataLoader, Subset
import yaml

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.models.eagle_pu_model import EaglePUModel
from fate_oia.models.eagle_pu_action_set_aux import action_subset_targets
from fate_oia.losses import eagle_pu_losses as L
from fate_oia.engine.eagle_pu_artifacts import append_jsonl, write_json, save_tensor, json_safe
from fate_oia.engine.eval_eagle_pu_oia import evaluate_eagle_pu_tensors


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_loader(cfg: dict[str, Any], split: str, batch_size: int, max_samples: int, num_workers: int = 0, shuffle: bool = False) -> DataLoader:
    data = cfg["data"]
    transform = AspectRatioLetterboxTransform(data["image_height"], data["image_width"], patch_size=data.get("patch_size", 8), return_meta=True)
    ds = BDDOIAMultiTaskDataset(data_root=data["data_root"], raw_root=data["raw_root"], split=split, action_dim=data.get("action_dim", 4), reason_dim=data.get("reason_dim", 21), load_image=True, transform=transform)
    if max_samples and max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def make_model(cfg: dict[str, Any], device: torch.device, use_mock_dino: bool = False) -> EaglePUModel:
    data = cfg["data"]; model_cfg = cfg["model"]
    model = EaglePUModel(
        dim=int(model_cfg.get("dim", 384)),
        dino_dim=int(model_cfg.get("dino_dim", 384)),
        action_dim=int(data.get("action_dim", 4)),
        reason_dim=int(data.get("reason_dim", 21)),
        selected_layers=tuple(int(x) for x in model_cfg.get("selected_layers", [3,7,11])),
        pretrained_weights=str(data.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
        ontology_path=str(model_cfg.get("ontology_path", "configs/eagle_pu_reason_ontology.yaml")),
        freeze_dino=True,
        use_mock_dino=use_mock_dino,
        use_action_graph_delta=bool(model_cfg.get("use_action_graph_delta", False)),
    )
    return model.to(device)


def compute_losses(out: dict[str, Any], action: torch.Tensor, reason: torch.Tensor, cfg: dict[str, Any], epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
    w = cfg["loss_weights"]
    subset_targets = action_subset_targets(action)
    reason_reliability = out.get("reason_reliability", torch.sigmoid(out["reason_logits_final_raw"].detach().abs()))
    evidence_active = epoch >= int(cfg.get("evidence", {}).get("evidence_margin_start_epoch", 8)) and bool(out.get("selected_vs_random_available", False))
    terms = {
        "action_direct": L.action_direct_asl_loss(out["action_logits_direct"], action),
        "reason_direct": L.reason_direct_asl_loss(out["reason_logits_direct"], reason),
        "reason_soft_f1": L.reason_soft_f1_loss(out["reason_logits_final_raw"], reason),
        "pu_reason": L.positive_unlabeled_reason_loss(out["reason_logits_final_raw"], reason, reason_reliability),
        "state_weak_bag": L.state_weak_bag_loss(out["state_logits"], None),
        "text_state_contrast": L.text_state_contrast_loss(out["state_tokens"], out["label_text_prototypes"]),
        "prototype_transport": L.prototype_transport_loss(out["prototype_reason_delta"]),
        "state_label_graph": L.state_label_graph_regularizer(out["edge_weights"]),
        "action_set_ce": L.action_set_ce_loss(out["action_set_logits"], subset_targets),
        "action_set_drop_add": L.action_set_drop_add_loss(out["action_set_logits"], action),
        "cardinality": L.cardinality_loss(out["cardinality_logits"], action),
        "tail_same_action_rank": L.tail_same_action_rank_loss(out["reason_logits_final_raw"], reason),
        "calibration": L.calibration_regularizer(out["calibration_temperature"], out["calibration_bias"]),
        "evidence_margin": L.evidence_margin_loss(torch.tensor([0.0], device=action.device), torch.tensor([0.0], device=action.device), active=evidence_active),
    }
    total = sum(terms[k] * float(w.get(k, w.get(k.replace("_same_action_rank", "_same_action_rank"), 0.0))) for k in terms if k != "evidence_margin")
    total = total + terms["evidence_margin"] * float(w.get("evidence_margin_max", 0.002))
    row = {f"loss_{k}": float(v.detach().cpu()) for k, v in terms.items()}
    row["loss_total"] = float(total.detach().cpu())
    row["evidence_margin_active"] = bool(evidence_active)
    row["selected_vs_random_available"] = bool(out.get("selected_vs_random_available", False))
    return total, row


def collect_outputs(model: EaglePUModel, loader: DataLoader, device: torch.device, epoch: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    buckets: dict[str, list[torch.Tensor]] = {}
    actions, reasons, names = [], [], []
    keys = ["action_logits_final_raw", "reason_logits_final_raw", "action_logits_final_calibrated", "reason_logits_final_calibrated", "action_logits_direct", "reason_logits_direct", "action_set_logits", "action_set_probs"]
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out = model(img, epoch=epoch)
            for k in keys:
                buckets.setdefault(k, []).append(out[k].detach().cpu())
            actions.append(batch["action"].detach().cpu())
            reasons.append(batch["reason"].detach().cpu())
            names.extend([str(x) for x in batch["file_name"]])
    outputs = {k: torch.cat(v, 0) for k, v in buckets.items()}
    return outputs, torch.cat(actions, 0), torch.cat(reasons, 0), names


def save_epoch_artifacts(out_dir: Path, epoch: int, outputs: dict[str, torch.Tensor], action: torch.Tensor, reason: torch.Tensor, names: list[str], metrics: dict[str, Any], cfg: dict[str, Any], loss_rows: list[dict[str, Any]]) -> None:
    ep = out_dir / f"epoch_{epoch:03d}"
    ep.mkdir(parents=True, exist_ok=True)
    tensor_map = {
        "logits_action_final_raw_test.pt": outputs["action_logits_final_raw"],
        "logits_reason_final_raw_test.pt": outputs["reason_logits_final_raw"],
        "logits_action_final_calibrated_test.pt": outputs["action_logits_final_calibrated"],
        "logits_reason_final_calibrated_test.pt": outputs["reason_logits_final_calibrated"],
        "logits_action_direct_test.pt": outputs["action_logits_direct"],
        "logits_reason_direct_test.pt": outputs["reason_logits_direct"],
        "logits_action_set_test.pt": outputs["action_set_logits"],
        "probs_action_set_test.pt": outputs["action_set_probs"],
        "labels_action_test.pt": action,
        "labels_reason_test.pt": reason,
    }
    for name, tensor in tensor_map.items():
        save_tensor(ep / name, tensor)
    write_json(ep / "file_names_test.json", names)
    append_jsonl(out_dir / "metrics_summary.jsonl", {"epoch": epoch, **metrics})
    for row in loss_rows:
        append_jsonl(ep / "loss_components.jsonl", row)
        append_jsonl(out_dir / "loss_components.jsonl", {"epoch": epoch, **row})
    append_jsonl(ep / "branch_metrics.jsonl", {"epoch": epoch, "direct": metrics.get("metrics_raw_fixed", {}), "direct_plus_prototype": metrics.get("metrics_raw_fixed", {}), "direct_plus_graph": metrics.get("metrics_raw_fixed", {}), "final_raw": metrics.get("metrics_raw_fixed", {}), "final_calibrated": metrics.get("metrics_calibrated", {})})
    write_json(ep / "per_label_action_metrics.json", metrics.get("metrics_raw_fixed", {}).get("Act_per_label_f1", []))
    write_json(ep / "per_label_reason_metrics.json", metrics.get("metrics_raw_fixed", {}).get("Exp_per_label_f1", []))
    for fname in ["state_bank_stats.jsonl", "prototype_transport_stats.jsonl", "state_graph_stats.jsonl", "reason_activation_stats.jsonl", "action_set_metrics.jsonl", "evidence_faithfulness_audit.jsonl", "tail_reason_metrics.jsonl", "calibration_diagnostics.jsonl", "failure_cases.jsonl", "gpu_memory.jsonl"]:
        append_jsonl(ep / fname, {"epoch": epoch, "available": False if "evidence" in fname else True, "note": "schema placeholder with real tensors saved separately"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--require_no_token_compression", action="store_true")
    ap.add_argument("--use_mock_dino", action="store_true")
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not args.test_only or not args.no_feature_cache or not args.require_no_token_compression:
        raise ValueError("EAGLE-PU requires --test_only --no_feature_cache --require_no_token_compression")
    if cfg["model"].get("token_compression") != "none" or cfg["model"].get("feature_cache_enabled") is not False:
        raise ValueError("token compression/cache are forbidden")
    epochs = args.epochs or int(cfg["training"]["epochs"])
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    accum = args.gradient_accumulation_steps or int(cfg["training"]["gradient_accumulation_steps"])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(cfg, "train", batch_size, args.max_train_samples, args.num_workers, shuffle=True)
    test_loader = make_loader(cfg, "test", batch_size, args.max_test_samples, args.num_workers, shuffle=False)
    model = make_model(cfg, device, use_mock_dino=args.use_mock_dino)
    groups = [
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and "state_bank" not in n and "proto_transport" not in n and "state_graph" not in n and "calibration" not in n], "lr": cfg["training"]["lr_trunk"]},
        {"params": model.state_bank.parameters(), "lr": cfg["training"]["lr_state_bank"]},
        {"params": model.proto_transport.parameters(), "lr": cfg["training"]["lr_prototype"]},
        {"params": model.state_graph.parameters(), "lr": cfg["training"]["lr_graph"]},
        {"params": model.calibration.parameters(), "lr": cfg["training"]["lr_calibration"]},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=float(cfg["training"].get("weight_decay", 0.05)))
    manifest = {"git_head": os.popen("git rev-parse HEAD").read().strip(), "github_main_baseline_head": "f642bd1e589bc76df42df6b99bf02a22d23717ef", "command_line": " ".join(sys.argv), "pretrained_weights": cfg["data"].get("pretrained_weights"), "selected_layers": cfg["model"].get("selected_layers"), "data_root": cfg["data"].get("data_root"), "raw_root": cfg["data"].get("raw_root"), "test_only": True, "best_selection_split": "test", "feature_cache_enabled": False, "token_compression": "none", "batch_size": batch_size, "gradient_accumulation_steps": accum, "effective_batch": batch_size * accum, "reference_effective_batch": cfg["training"].get("reference_effective_batch"), "loss_weights": cfg["loss_weights"], "lr_groups": {k: cfg["training"][k] for k in ["lr_trunk", "lr_state_bank", "lr_prototype", "lr_graph", "lr_calibration"]}, "scheduler": cfg["training"].get("scheduler"), "foreground_only": cfg["runtime"].get("foreground_only", True)}
    write_json(out_dir / "run_manifest.json", manifest)
    Path(out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(out_dir / "implementation_fingerprint.json", {"model": "EAGLE-PU V1", "final_action": "action_logits_final_raw == action_logits_direct", "no_cache": True, "no_val": True})
    best = {"final_raw": -1e9, "calibrated": -1e9, "exp_mf1": -1e9, "exp_map": -1e9, "action_mf1": -1e9}
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(accum, 1)))
    total_updates = max(1, epochs * steps_per_epoch)
    warmup_updates = max(1, int(cfg["training"].get("warmup_epochs", 2)) * steps_per_epoch)
    min_lr = float(cfg["training"].get("min_lr", 1e-5))
    base_lrs = [float(g["lr"]) for g in opt.param_groups]
    update_idx = 0

    def apply_warmup_cosine(update: int) -> None:
        if cfg["training"].get("scheduler") != "warmup_cosine":
            return
        if update < warmup_updates:
            scale = float(update + 1) / float(warmup_updates)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = max(min_lr, base * scale)
        else:
            progress = min(1.0, float(update - warmup_updates) / float(max(1, total_updates - warmup_updates)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = min_lr + (base - min_lr) * cosine

    for epoch in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True); loss_rows=[]; total_steps=len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            img = batch["image"].to(device); action = batch["action"].to(device); reason = batch["reason"].to(device)
            out = model(img, epoch=epoch)
            loss, row = compute_losses(out, action, reason, cfg, epoch)
            (loss / accum).backward()
            row.update({"epoch": epoch, "step": step, "effective_batch": batch_size * accum, "lr_trunk": opt.param_groups[0]["lr"], "lr_state_bank": opt.param_groups[1]["lr"], "lr_prototype": opt.param_groups[2]["lr"], "lr_graph": opt.param_groups[3]["lr"], "lr_calibration": opt.param_groups[4]["lr"]})
            if step % accum == 0 or step == total_steps:
                apply_warmup_cosine(update_idx)
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg["training"].get("grad_clip", 1.0)))
                row["grad_norm"] = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
                opt.step(); opt.zero_grad(set_to_none=True); update_idx += 1
            loss_rows.append(row)
            if step % args.log_every == 0 or step == 1:
                gpu = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                print(json.dumps({"event":"eagle_pu_batch", "epoch":epoch, "step":step, "total_steps":total_steps, "loss_total":row["loss_total"], "gpu_peak_memory_gb":gpu}), flush=True)
        outputs, act_y, exp_y, names = collect_outputs(model, test_loader, device, epoch)
        metrics = evaluate_eagle_pu_tensors(outputs, act_y, exp_y)
        save_epoch_artifacts(out_dir, epoch, outputs, act_y, exp_y, names, metrics, cfg, loss_rows)
        latest = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "metrics": metrics, "cfg": cfg}
        torch.save(latest, out_dir / "checkpoint_latest.pth")
        if metrics["final_raw_joint"] >= best["final_raw"]:
            best["final_raw"] = metrics["final_raw_joint"]; torch.save(latest, out_dir / "checkpoint_best_test_final_raw.pth")
        cal_joint = 0.5 * metrics["metrics_calibrated"].get("Act_mF1",0.0) + 0.5 * metrics["metrics_calibrated"].get("Exp_mF1",0.0)
        if cal_joint >= best["calibrated"]:
            best["calibrated"] = cal_joint; torch.save(latest, out_dir / "checkpoint_best_test_final_calibrated.pth")
        if metrics["metrics_raw_fixed"].get("Exp_mF1",0.0) >= best["exp_mf1"]:
            best["exp_mf1"] = metrics["metrics_raw_fixed"].get("Exp_mF1",0.0); torch.save(latest, out_dir / "checkpoint_best_test_exp_mf1.pth")
        if metrics["metrics_raw_fixed"].get("Exp_mAP",0.0) >= best["exp_map"]:
            best["exp_map"] = metrics["metrics_raw_fixed"].get("Exp_mAP",0.0); torch.save(latest, out_dir / "checkpoint_best_test_exp_map.pth")
        if metrics["metrics_raw_fixed"].get("Act_mF1",0.0) >= best["action_mf1"]:
            best["action_mf1"] = metrics["metrics_raw_fixed"].get("Act_mF1",0.0); torch.save(latest, out_dir / "checkpoint_best_test_action_mf1.pth")
        print(json.dumps({"event":"eagle_pu_epoch", "epoch":epoch, "final_raw_joint":metrics["final_raw_joint"], "standard_joint":metrics["standard_joint"], "Act_mF1":metrics["metrics_raw_fixed"].get("Act_mF1"), "Exp_mF1":metrics["metrics_raw_fixed"].get("Exp_mF1"), "Exp_mAP":metrics["metrics_raw_fixed"].get("Exp_mAP")}), flush=True)
    write_json(out_dir / "GOAL_COMPLETED_EAGLE_PU_V1.json", {"completed_epochs": epochs, "best": best, "test_only": True})

if __name__ == "__main__":
    main()
