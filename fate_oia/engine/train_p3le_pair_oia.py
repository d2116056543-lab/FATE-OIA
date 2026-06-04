from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.engine.train_fate_oia import build_backbone, build_transform, extract_tokens
from fate_oia.losses.cagrad_lite import clip_shared_gradient_budget
from fate_oia.losses.p3le_pair_losses import p3le_pair_loss
from fate_oia.models.p3le_pair_oia_model import P3LEPairOIAFeatureModel
from fate_oia.utils.p3le_pair_artifacts import append_jsonl, save_logits_artifacts, summarize_scalar_rows, write_json


def load_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    out: dict[str, Any] = {}
    for value in data.values():
        if isinstance(value, dict):
            out.update(value)
    out.update({k: v for k, v in data.items() if not isinstance(v, dict)})
    return out


def apply_config(args: argparse.Namespace, defaults: dict[str, Any], config: dict[str, Any]) -> None:
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == defaults.get(key):
            setattr(args, key, value)


def limit_dataset(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        return Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def make_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    dataset = BDDOIAMultiTaskDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        split=split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=True,
        transform=build_transform(args.image_height, args.image_width, args.patch_size, args.preserve_aspect_ratio, return_meta=True),
    )
    dataset = limit_dataset(dataset, args.max_train_samples if split == "train" else args.max_test_samples)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def combined(action_logits: torch.Tensor, reason_logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([action_logits, reason_logits], dim=1)


def metrics_for(action_logits: torch.Tensor, reason_logits: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    result = evaluate_snna25(combined(action_logits, reason_logits), labels, args.action_dim, "fixed", args.eval_threshold)
    return result["metrics"]


def joint_from_metrics(metrics: dict[str, Any]) -> float:
    return 0.5 * float(metrics["Act_mF1"]) + 0.5 * float(metrics["Exp_mF1"])


def tail_metrics(metrics: dict[str, Any], tail_indices: list[int]) -> dict[str, Any]:
    f1 = metrics.get("Exp_per_label_f1", [])
    ap = metrics.get("Exp_per_label_ap", [])
    valid = [idx for idx in tail_indices if idx < len(f1)]
    return {
        "tail_indices": valid,
        "tail_Exp_mF1": float(sum(float(f1[idx]) for idx in valid) / len(valid)) if valid else 0.0,
        "tail_Exp_mAP": float(sum(float(ap[idx]) for idx in valid) / len(valid)) if valid else 0.0,
    }


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def run_epoch(args, backbone, model, loader, optimizer, scheduler, device, epoch: int, train: bool) -> dict[str, Any]:
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    all_labels: list[torch.Tensor] = []
    output_lists: dict[str, list[torch.Tensor]] = {
        "base_action_logits": [],
        "base_reason_logits": [],
        "action_specialist_logits": [],
        "reason_specialist_logits": [],
        "final_action_logits": [],
        "final_reason_logits": [],
        "action_set_logits": [],
    }
    file_names: list[str] = []
    losses: list[dict[str, float]] = []
    total_loss = 0.0
    total_count = 0
    accum = max(1, int(args.gradient_accumulation_steps))
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            reason = batch["reason"].to(device, non_blocking=True)
            labels = torch.cat([action, reason], dim=1)
            tokens = extract_tokens(backbone, images, args.n_last_blocks)
            outputs = model(tokens, action_labels=action, reason_labels=reason, epoch=epoch)
            loss, parts = p3le_pair_loss(outputs, action, reason, args)
            if train:
                (loss / float(accum)).backward()
                if ((step + 1) % accum == 0) or ((step + 1) == len(loader)):
                    shared_norm = clip_shared_gradient_budget(model.shared_parameters_for_budget(), args.shared_grad_budget_norm)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    parts["shared_grad_norm_before_budget"] = float(shared_norm)
            batch_size = int(images.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_count += batch_size
            parts.update({"epoch": epoch, "step": step, "train": bool(train), "lr": current_lr(optimizer), "batch_size": batch_size})
            losses.append(parts)
            if train and step % int(args.log_every) == 0:
                print(json.dumps({"event": "p3le_pair_batch", **parts}, ensure_ascii=False), flush=True)
            if not train:
                all_labels.append(labels.detach().cpu())
                for key in output_lists:
                    output_lists[key].append(outputs[key].detach().cpu())
                fn = batch.get("file_name", [])
                file_names.extend([str(x) for x in (fn if not isinstance(fn, str) else [fn])])
    labels_all = torch.cat(all_labels, 0) if all_labels else torch.empty(0, args.action_dim + args.reason_dim)
    outputs_all = {key: torch.cat(value, 0) if value else torch.empty(0, args.action_dim if "action" in key else args.reason_dim) for key, value in output_lists.items()}
    metrics = {}
    branch_metrics = {}
    if not train and labels_all.numel() > 0:
        branch_defs = {
            "base": ("base_action_logits", "base_reason_logits"),
            "action_specialist": ("action_specialist_logits", "base_reason_logits"),
            "reason_specialist": ("base_action_logits", "reason_specialist_logits"),
            "action_set": ("action_set_logits", "base_reason_logits"),
            "final": ("final_action_logits", "final_reason_logits"),
        }
        for name, (ak, rk) in branch_defs.items():
            branch_metrics[name] = metrics_for(outputs_all[ak], outputs_all[rk], labels_all, args)
        metrics = branch_metrics["final"]
    return {
        "loss": total_loss / max(1, total_count),
        "loss_rows": losses,
        "labels": labels_all,
        "outputs": outputs_all,
        "file_names": file_names,
        "metrics": metrics,
        "branch_metrics": branch_metrics,
        "joint": joint_from_metrics(metrics) if metrics else 0.0,
    }


def save_epoch(args, out_dir: Path, epoch: int, train_stats: dict[str, Any], test_stats: dict[str, Any], manifest: dict[str, Any]) -> None:
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for row in train_stats["loss_rows"]:
        append_jsonl(epoch_dir / "loss_components.jsonl", row)
    branch_metrics = test_stats["branch_metrics"]
    write_json(epoch_dir / "branch_metrics.json", branch_metrics)
    write_json(epoch_dir / "metrics_summary.json", {"epoch": epoch, "test_metrics": test_stats["metrics"], "joint_test_score": test_stats["joint"]})
    final_metrics = test_stats["metrics"]
    per_label = {
        "Exp_per_label_f1": final_metrics.get("Exp_per_label_f1", []),
        "Exp_per_label_ap": final_metrics.get("Exp_per_label_ap", []),
        "Act_per_label_f1": final_metrics.get("Act_per_label_f1", []),
        "Act_per_label_ap": final_metrics.get("Act_per_label_ap", []),
    }
    write_json(epoch_dir / "per_label_reason_metrics.json", per_label)
    tail = tail_metrics(final_metrics, args.tail_indices)
    write_json(epoch_dir / "tail_group_metrics.json", tail)
    loss_summary = summarize_scalar_rows(train_stats["loss_rows"])
    write_json(epoch_dir / "pair_tensor_stats.json", {k: loss_summary.get(k, 0.0) for k in ["pair_tensor_mean", "pair_tensor_std", "pair_seed_loss", "pair_consistency_loss"]})
    write_json(epoch_dir / "reason_reliability_stats.json", {k: loss_summary.get(k, 0.0) for k in ["q_mean", "q_min", "q_max", "q_entropy"]})
    write_json(epoch_dir / "evidence_bag_stats.json", {k: loss_summary.get(k, 0.0) for k in ["evidence_bag_loss", "evidence_selected_mean", "evidence_random_mean", "evidence_lambda_active"]})
    write_json(epoch_dir / "selected_vs_random_evidence_stats.json", {k: loss_summary.get(k, 0.0) for k in ["evidence_selected_mean", "evidence_random_mean", "evidence_lambda_active"]})
    write_json(epoch_dir / "router_stats.json", {k: loss_summary.get(k, 0.0) for k in ["router_scale", "action_gate_mean", "reason_gate_mean", "pareto_action_loss", "pareto_reason_loss"]})
    write_json(epoch_dir / "action_set_stats.json", {k: loss_summary.get(k, 0.0) for k in ["action_set_loss"]})
    (epoch_dir / "failure_cases.jsonl").write_text("", encoding="utf-8")
    visual_dir = epoch_dir / "visual_samples"
    visual_dir.mkdir(exist_ok=True)
    append_jsonl(visual_dir / "fate_snna_schema.jsonl", {"epoch": epoch, "note": "schema placeholder; visual export is diagnostic-only for P3LE-PAIR V1"})
    save_logits_artifacts(epoch_dir, test_stats["outputs"], test_stats["labels"], test_stats["file_names"], args.action_dim)
    for name, data in [
        ("branch_metrics.jsonl", {"epoch": epoch, "branch_metrics": branch_metrics}),
        ("router_stats.jsonl", {"epoch": epoch, **(json.loads((epoch_dir / "router_stats.json").read_text(encoding="utf-8")))}),
        ("tail_metrics.jsonl", {"epoch": epoch, **tail}),
        ("pair_stats.jsonl", {"epoch": epoch, **(json.loads((epoch_dir / "pair_tensor_stats.json").read_text(encoding="utf-8")))}),
        ("reliability_stats.jsonl", {"epoch": epoch, **(json.loads((epoch_dir / "reason_reliability_stats.json").read_text(encoding="utf-8")))}),
        ("selected_vs_random_evidence.jsonl", {"epoch": epoch, **(json.loads((epoch_dir / "selected_vs_random_evidence_stats.json").read_text(encoding="utf-8")))}),
    ]:
        append_jsonl(out_dir / name, data)
    write_json(epoch_dir / "run_manifest.json", manifest)


def build_manifest(args, out_dir: Path, train_count: int, test_count: int) -> dict[str, Any]:
    return {
        "repo_name": "FATE-OIA",
        "method": "P3LE-PAIR-OIA V1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "data_root": args.data_root,
        "raw_root": args.raw_root,
        "bdd100k_root": args.bdd100k_root,
        "pretrained_weights": args.pretrained_weights,
        "feature_cache_enabled": False,
        "token_compression": "none",
        "eval_splits": ["test"],
        "best_selection_split": "test",
        "best_selection_metric": "joint_test_score",
        "train_split_count": int(train_count),
        "test_split_count": int(test_count),
        "output_dir": str(out_dir),
        "config_resolved": vars(args),
    }


def build_scheduler(args, optimizer):
    def lr_lambda(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        progress = (epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        min_ratio = args.min_lr / max(args.lr, 1e-12)
        return min_ratio + (1.0 - min_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train P3LE-PAIR-OIA V1 with direct BDD-OIA images and test-only eval.")
    ap.add_argument("--config", default="")
    ap.add_argument("--data_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--raw_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--bdd100k_root", default="E:/sbw/BDD100K")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pretrained_weights", default="E:/sbw/SNNA_repro/SNNA/ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--shared_lr", type=float, default=2e-4)
    ap.add_argument("--router_lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--router_weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--shared_grad_budget_norm", type=float, default=1.0)
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--loss_action_gt", type=float, default=1.0)
    ap.add_argument("--loss_reason_gt", type=float, default=1.0)
    ap.add_argument("--loss_a_action", type=float, default=0.5)
    ap.add_argument("--loss_r_reason", type=float, default=0.5)
    ap.add_argument("--loss_a_reason", type=float, default=0.05)
    ap.add_argument("--loss_r_action", type=float, default=0.0)
    ap.add_argument("--loss_action_set", type=float, default=0.1)
    ap.add_argument("--loss_pair_seed", type=float, default=0.05)
    ap.add_argument("--loss_pair_consistency", type=float, default=0.02)
    ap.add_argument("--loss_evidence_bag", type=float, default=0.01)
    ap.add_argument("--loss_q_entropy", type=float, default=0.001)
    ap.add_argument("--loss_pareto", type=float, default=0.1)
    ap.add_argument("--pareto_margin_action", type=float, default=0.005)
    ap.add_argument("--pareto_margin_reason", type=float, default=0.005)
    ap.add_argument("--action_residual_cap", type=float, default=0.04)
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--tail_indices", type=int, nargs="*", default=[5, 6, 9, 10, 11, 12, 13, 14])
    ap.add_argument("--feature_cache", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--test_only_evaluation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--token_compression", default="none")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=120)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    defaults = vars(ap.parse_args(["--output_dir", "__dummy__"]))
    apply_config(args, defaults, load_config(args.config))
    if args.feature_cache:
        raise ValueError("P3LE-PAIR-OIA V1 forbids feature cache")
    if not args.test_only_evaluation:
        raise ValueError("P3LE-PAIR-OIA V1 must use test-only evaluation")
    if str(args.token_compression).lower() != "none":
        raise ValueError("P3LE-PAIR-OIA V1 formal run forbids token compression")
    if not Path(args.pretrained_weights).exists():
        raise FileNotFoundError(f"pretrained_weights does not exist: {args.pretrained_weights}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "args.json", vars(args))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    backbone, dim = build_backbone(args, device)
    model = P3LEPairOIAFeatureModel(dim, args.action_dim, args.reason_dim, tail_indices=tuple(args.tail_indices), action_residual_cap=args.action_residual_cap).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.shared_encoder.parameters()) + list(model.ple.shared_1.parameters()) + list(model.ple.shared_2.parameters()), "lr": args.shared_lr, "weight_decay": args.weight_decay},
            {"params": list(model.ple.action_1.parameters()) + list(model.ple.action_2.parameters()) + list(model.action_head.parameters()) + list(model.action_set_head.parameters()), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": list(model.ple.reason_1.parameters()) + list(model.ple.reason_2.parameters()) + list(model.ple.tail_2.parameters()) + list(model.reason_head.parameters()) + list(model.pair_head.parameters()) + list(model.reliability.parameters()), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": list(model.router.parameters()), "lr": args.router_lr, "weight_decay": args.router_weight_decay},
            {"params": list(model.evidence_bag.parameters()) + list(model.action_aux_reason.parameters()) + list(model.reason_aux_action.parameters()), "lr": args.lr, "weight_decay": args.weight_decay},
        ]
    )
    scheduler = build_scheduler(args, optimizer)
    train_loader = make_loader(args, "train", True)
    test_loader = make_loader(args, "test", False)
    manifest = build_manifest(args, out_dir, len(train_loader.dataset), len(test_loader.dataset))
    write_json(out_dir / "run_manifest.json", manifest)
    best_score = -1.0
    for epoch in range(args.epochs):
        train_stats = run_epoch(args, backbone, model, train_loader, optimizer, scheduler, device, epoch, True)
        test_stats = run_epoch(args, backbone, model, test_loader, optimizer, scheduler, device, epoch, False)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "test_loss": test_stats["loss"],
            "joint_test_score": test_stats["joint"],
            "test_metrics": test_stats["metrics"],
            "branch_metrics": test_stats["branch_metrics"],
            "lr": current_lr(optimizer),
            "best_selection_split": "test",
        }
        append_jsonl(out_dir / "metrics_summary.jsonl", row)
        write_json(out_dir / "metrics_summary.json", row)
        save_epoch(args, out_dir, epoch, train_stats, test_stats, manifest)
        ckpt = {"epoch": epoch, "method": "P3LE-PAIR-OIA V1", "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "args": vars(args), "best_test_score": max(best_score, test_stats["joint"])}
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        if test_stats["joint"] >= best_score:
            best_score = test_stats["joint"]
            torch.save(ckpt, out_dir / "checkpoint_best_test.pth")
            torch.save(ckpt, out_dir / "checkpoint_best.pth")
            write_json(out_dir / "metrics_best_test.json", row)
        print(json.dumps({"event": "p3le_pair_epoch", **row}, ensure_ascii=False), flush=True)
    write_json(out_dir / "GOAL_COMPLETED_P3LE_PAIR_OIA_V1.json", {"best_test_score": best_score, "output_dir": str(out_dir), "test_only": True})


if __name__ == "__main__":
    main()
