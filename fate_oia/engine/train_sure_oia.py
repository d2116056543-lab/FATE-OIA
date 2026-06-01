from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.sure_oia_dataset import SUREOIADataset, sure_collate
from fate_oia.engine.train_fate_oia import build_backbone, extract_tokens, load_config_defaults, apply_config_defaults
from fate_oia.losses.gradnorm import GradNormBalancer
from fate_oia.losses.sure_losses import compute_sure_losses, make_sure_criterion
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.sure_oia_model import SUREOIAFeatureModel
from fate_oia.utils.sure_artifacts import append_jsonl, build_run_manifest, write_json


def _limited(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        return Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def make_sure_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    dataset = SUREOIADataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        bdd100k_root=args.bdd100k_root,
        split=split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        image_height=args.image_height,
        image_width=args.image_width,
        patch_size=args.patch_size,
        preserve_aspect_ratio=args.preserve_aspect_ratio,
    )
    dataset = _limited(dataset, args.max_train_samples if split == "train" else args.max_test_samples)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=sure_collate)


def _to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    action = batch["action"].to(device, non_blocking=True).float()
    reason = batch["reason"].to(device, non_blocking=True).float()
    return images, action, reason


def _joint(metrics: dict[str, Any]) -> float:
    return 0.5 * float(metrics.get("Act_mF1_final", 0.0)) + 0.5 * float(metrics.get("Exp_mF1_final", 0.0))


def _summarize_relation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"candidate_edges": 0, "selected_edges": 0, "selected_ratio": 0.0}
    cand = sum(int(r.get("candidate_edges", 0)) for r in rows)
    sel = sum(int(r.get("selected_edges", 0)) for r in rows)
    return {"candidate_edges": cand, "selected_edges": sel, "selected_ratio": sel / max(cand, 1), "batches": len(rows)}


def evaluate(args, backbone: nn.Module, model: nn.Module, loader: DataLoader, device: torch.device, epoch_dir: Path) -> dict[str, Any]:
    model.eval()
    action_final, action_base, action_upper = [], [], []
    reason_final, reason_base, reason_upper = [], [], []
    labels_action, labels_reason, file_names = [], [], []
    relation_rows: list[dict[str, Any]] = []
    memory_gates: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images, action, reason = _to_device(batch, device)
            tokens = extract_tokens(backbone, images, args.n_last_blocks)
            out = model(tokens, structured=batch.get("bdd100k_structured"), image_meta=batch.get("image_meta"), return_gt_scene_upper=True)
            action_final.append(out["action_final_logits"].detach().cpu())
            action_base.append(out["action_base_logits"].detach().cpu())
            reason_final.append(out["reason_final_logits"].detach().cpu())
            reason_base.append(out["reason_base_logits"].detach().cpu())
            action_upper.append(out.get("action_gt_scene_upper_logits", torch.empty(action.shape[0], args.action_dim, device=device)).detach().cpu())
            reason_upper.append(out.get("reason_gt_scene_upper_logits", torch.empty(reason.shape[0], args.reason_dim, device=device)).detach().cpu())
            labels_action.append(action.detach().cpu())
            labels_reason.append(reason.detach().cpu())
            file_names.extend(batch.get("file_name", []))
            relation_rows.append(out.get("relation_stats", {}))
            if "memory_gate" in out:
                memory_gates.extend([float(x) for x in out["memory_gate"].detach().cpu().flatten()])
    af = torch.cat(action_final, 0)
    ab = torch.cat(action_base, 0)
    au = torch.cat(action_upper, 0)
    rf = torch.cat(reason_final, 0)
    rb = torch.cat(reason_base, 0)
    ru = torch.cat(reason_upper, 0)
    ya = torch.cat(labels_action, 0)
    yr = torch.cat(labels_reason, 0)
    metrics_final_a = multilabel_metrics_from_logits(af, ya, args.eval_threshold, prefix="Act_")
    metrics_final_r = multilabel_metrics_from_logits(rf, yr, args.eval_threshold, prefix="Exp_")
    metrics_base_a = multilabel_metrics_from_logits(ab, ya, args.eval_threshold, prefix="ActBase_")
    metrics_base_r = multilabel_metrics_from_logits(rb, yr, args.eval_threshold, prefix="ExpBase_")
    metrics_upper_a = multilabel_metrics_from_logits(au, ya, args.eval_threshold, prefix="ActUpper_")
    metrics_upper_r = multilabel_metrics_from_logits(ru, yr, args.eval_threshold, prefix="ExpUpper_")
    metrics = {
        "Act_mF1_final": metrics_final_a["Act_mF1"],
        "Act_oF1_final": metrics_final_a["Act_oF1"],
        "Act_mAP_final": metrics_final_a["Act_mAP"],
        "Exp_mF1_final": metrics_final_r["Exp_mF1"],
        "Exp_oF1_final": metrics_final_r["Exp_oF1"],
        "Exp_mAP_final": metrics_final_r["Exp_mAP"],
        "Act_mF1_base": metrics_base_a["ActBase_mF1"],
        "Exp_mF1_base": metrics_base_r["ExpBase_mF1"],
        "Act_mF1_gt_scene_upper": metrics_upper_a["ActUpper_mF1"],
        "Exp_mF1_gt_scene_upper": metrics_upper_r["ExpUpper_mF1"],
        "per_reason_f1_final": metrics_final_r["Exp_per_label_f1"],
        "per_reason_ap_final": metrics_final_r["Exp_per_label_ap"],
    }
    metrics["joint_test_score"] = _joint(metrics)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    logits_dir = epoch_dir / "logits"
    logits_dir.mkdir(exist_ok=True)
    torch.save(af, logits_dir / "action_final_test.pt")
    torch.save(ab, logits_dir / "action_base_test.pt")
    torch.save(au, logits_dir / "action_gt_scene_upper_test.pt")
    torch.save(rf, logits_dir / "reason_final_test.pt")
    torch.save(rb, logits_dir / "reason_base_test.pt")
    torch.save(ru, logits_dir / "reason_gt_scene_upper_test.pt")
    torch.save(ya, logits_dir / "labels_action_test.pt")
    torch.save(yr, logits_dir / "labels_reason_test.pt")
    write_json(logits_dir / "file_names_test.json", file_names)
    relation_summary = _summarize_relation(relation_rows)
    action_safe = {"memory_gate_mean": float(sum(memory_gates) / max(len(memory_gates), 1)), "memory_gate_count": len(memory_gates)}
    write_json(epoch_dir / "metrics_summary.json", metrics)
    write_json(epoch_dir / "relation_stats.json", relation_summary)
    write_json(epoch_dir / "action_safe_stats.json", action_safe)
    write_json(epoch_dir / "branch_metrics.json", {**metrics_base_a, **metrics_base_r, **metrics_upper_a, **metrics_upper_r})
    append_jsonl(epoch_dir / "sure_visuals_index.jsonl", {"epoch": int(epoch_dir.name.split("_")[-1]), "schema_only": True, "file_count": len(file_names[:8])})
    return {"metrics": metrics, "relation_stats": relation_summary, "action_safe_stats": action_safe}


def train_epoch(args, backbone: nn.Module, model: nn.Module, balancer: GradNormBalancer, loader: DataLoader, optimizer, criterion, device: torch.device, epoch: int, out_dir: Path) -> dict[str, Any]:
    model.train()
    balancer.train()
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {}
    batch_count = 0
    start = time.time()
    for step, batch in enumerate(loader):
        images, action, reason = _to_device(batch, device)
        with torch.no_grad():
            tokens = extract_tokens(backbone, images, args.n_last_blocks)
        out = model(tokens, structured=batch.get("bdd100k_structured"), image_meta=batch.get("image_meta"), return_gt_scene_upper=True)
        losses = compute_sure_losses(out, action, reason, criterion, relation_teacher_weight=args.relation_teacher_weight)
        task_loss, grad_stats = balancer(losses)
        aux_loss = args.base_aux_weight * (losses["base_action"] + losses["base_reason"])
        total_loss = task_loss + aux_loss
        (total_loss / args.gradient_accumulation_steps).backward()
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(balancer.parameters()), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        batch_count += 1
        for key, value in {**losses, "total": total_loss}.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        if step % max(1, args.log_every) == 0:
            msg = {
                "event": "sure_train_batch",
                "epoch": epoch,
                "step": step,
                "batches": len(loader),
                "loss": float(total_loss.detach().cpu()),
                "action_loss": float(losses["action"].detach().cpu()),
                "reason_loss": float(losses["reason"].detach().cpu()),
                "selected_edges": int(out.get("relation_stats", {}).get("selected_edges", 0)),
                "memory_gate_mean": float(out.get("memory_gate", torch.zeros(1)).detach().mean().cpu()),
            }
            print(json.dumps(msg, ensure_ascii=False), flush=True)
    stats = {k: v / max(batch_count, 1) for k, v in totals.items()}
    stats["epoch_seconds"] = time.time() - start
    stats["batch_count"] = batch_count
    stats["gradnorm"] = grad_stats
    append_jsonl(out_dir / "loss_components.jsonl", {"epoch": epoch, **stats})
    return stats


def save_checkpoint(path: Path, epoch: int, model: nn.Module, balancer: GradNormBalancer, optimizer, best_score: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "balancer": balancer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_test_score": best_score,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, balancer: GradNormBalancer, optimizer) -> tuple[int, float]:
    if not path:
        return 0, -1.0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if "balancer" in ckpt:
        balancer.load_state_dict(ckpt["balancer"], strict=False)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", -1)) + 1, float(ckpt.get("best_test_score", -1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SURE-OIA v2 direct-image model with test-only evaluation.")
    ap.add_argument("--config", default="")
    ap.add_argument("--data_root", default="E:/sbw/BDD-OIA/data")
    ap.add_argument("--raw_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--bdd100k_root", default="E:/sbw/BDD100K")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--loss", choices=["asl", "bce"], default="asl")
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--relation_queries", type=int, default=32)
    ap.add_argument("--max_edges_per_label", type=int, default=8)
    ap.add_argument("--max_edges_total", type=int, default=96)
    ap.add_argument("--relation_teacher_weight", type=float, default=0.05)
    ap.add_argument("--base_aux_weight", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--eval_splits", default="test")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    apply_config_defaults(args, load_config_defaults(args.config))
    if str(args.eval_splits).strip() != "test":
        raise ValueError("SURE-OIA v2 plan requires --eval_splits test only.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "args.json", vars(args))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    backbone, dim = build_backbone(args, device)
    model = SUREOIAFeatureModel(dim=dim, action_dim=args.action_dim, reason_dim=args.reason_dim, relation_queries=args.relation_queries, max_edges_per_label=args.max_edges_per_label, max_edges_total=args.max_edges_total).to(device)
    balancer = GradNormBalancer().to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(balancer.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = make_sure_criterion(args.loss, args.asl_gamma_pos, args.asl_gamma_neg, args.asl_clip)
    train_loader = make_sure_loader(args, "train", True)
    test_loader = make_sure_loader(args, "test", False)
    manifest = build_run_manifest(args, Path.cwd(), len(train_loader.dataset), len(test_loader.dataset))
    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "training_config_resolved.yaml", vars(args))
    start_epoch, best_test = load_checkpoint(args.resume, model, balancer, optimizer)
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_epoch(args, backbone, model, balancer, train_loader, optimizer, criterion, device, epoch, out_dir)
        epoch_dir = out_dir / f"epoch_{epoch:03d}"
        test_stats = evaluate(args, backbone, model, test_loader, device, epoch_dir)
        metrics = test_stats["metrics"]
        row = {"epoch": epoch, "split": "test", **metrics, "train_loss": train_stats.get("total", 0.0)}
        append_jsonl(out_dir / "metrics_summary.jsonl", row)
        write_json(out_dir / "metrics_latest.json", row)
        write_json(epoch_dir / "gradnorm_stats.json", train_stats.get("gradnorm", {}))
        write_json(epoch_dir / "loss_components_summary.json", train_stats)
        append_jsonl(epoch_dir / "failure_cases.jsonl", {"epoch": epoch, "schema_only": True, "note": "Full failure table is generated from saved logits in downstream audit."})
        save_checkpoint(out_dir / "checkpoint_latest.pth", epoch, model, balancer, optimizer, best_test, args)
        if metrics["joint_test_score"] > best_test:
            best_test = float(metrics["joint_test_score"])
            save_checkpoint(out_dir / "checkpoint_best_test.pth", epoch, model, balancer, optimizer, best_test, args)
            write_json(out_dir / "metrics_best_test.json", row)
        history.append(row)
        print(json.dumps({"event": "sure_epoch_complete", "epoch": epoch, "joint_test_score": metrics["joint_test_score"], "Act_mF1": metrics["Act_mF1_final"], "Exp_mF1": metrics["Exp_mF1_final"], "best_test": best_test}, ensure_ascii=False), flush=True)
    write_json(out_dir / "GOAL_COMPLETED_SURE_OIA_V2.json", {"epochs": args.epochs, "best_test_score": best_test, "history_len": len(history), "test_only": True})


if __name__ == "__main__":
    main()
