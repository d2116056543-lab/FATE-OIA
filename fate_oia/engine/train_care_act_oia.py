from __future__ import annotations

import argparse
import math
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.care_moe_oia_dataset import CAREMoEOIADataset, care_moe_collate
from fate_oia.engine.eval_care_act_oia import select_guarded_action_branch
from fate_oia.engine.train_fate_oia import build_backbone, extract_tokens
from fate_oia.losses.care_act_losses import care_act_training_loss
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.care_act_model import CAREActOIAModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.care_act_artifacts import append_jsonl, write_json
from fate_oia.utils.care_act_review_gates import require_review_pass


def _load_config(args: argparse.Namespace) -> None:
    if not args.config:
        return
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    argv = set(sys.argv[1:])
    for k, v in cfg.items():
        if f"--{k}" not in argv and hasattr(args, k):
            setattr(args, k, v)


def _limit(ds, n: int):
    return Subset(ds, range(min(n, len(ds)))) if n and n > 0 else ds


def is_full_goal_run(args: argparse.Namespace) -> bool:
    return (
        args.epochs >= 24
        and args.max_train_samples <= 0
        and args.max_test_samples <= 0
        and args.test_only_evaluation
        and args.best_selection_split == "test"
    )


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "feature_cache_enabled", False):
        raise RuntimeError("CARE-ACT requires feature_cache_enabled=false")
    if getattr(args, "token_compression", "none") != "none":
        raise RuntimeError("CARE-ACT requires token_compression=none")
    if not args.test_only_evaluation or args.best_selection_split != "test":
        raise RuntimeError("CARE-ACT requires test-only evaluation and test best selection")


def make_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    transform = AspectRatioLetterboxTransform(args.image_height, args.image_width, patch_size=args.patch_size, return_meta=True)
    ds = CAREMoEOIADataset(
        data_root=args.data_root,
        raw_root=args.bdd_oia_root,
        bdd100k_root=args.bdd100k_root,
        split=split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=True,
        transform=transform,
        include_structured=(split == "train"),
    )
    ds = _limit(ds, args.max_train_samples if split == "train" else args.max_test_samples)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=care_moe_collate)


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k]))]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


@torch.no_grad()
def evaluate(args: argparse.Namespace, backbone, model: CAREActOIAModel, loader: DataLoader, device: torch.device, epoch: int, out_dir: Path) -> dict[str, Any]:
    model.eval()
    tensors: dict[str, list[torch.Tensor]] = {k: [] for k in [
        "action_base", "action_evidence", "action_set", "action_candidate", "reason_base", "reason_final", "labels_action", "labels_reason"
    ]}
    file_names: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        action = batch["action"].to(device)
        reason = batch["reason"].to(device)
        tokens = extract_tokens(backbone, images, args.n_last_blocks)
        out = model(tokens, batch=None, structured=None, epoch=epoch, force_structured_upper=False)
        tensors["action_base"].append(out["action_base_logits"].detach().cpu())
        tensors["action_evidence"].append(out["action_evidence_logits"].detach().cpu())
        tensors["action_set"].append(out["action_set_logits"].detach().cpu())
        tensors["action_candidate"].append(out["action_final_candidate_logits"].detach().cpu())
        tensors["reason_base"].append(out["reason_base_logits"].detach().cpu())
        tensors["reason_final"].append(out["reason_logits"].detach().cpu())
        tensors["labels_action"].append(action.detach().cpu())
        tensors["labels_reason"].append(reason.detach().cpu())
        file_names.extend(batch["file_name"])
        diagnostics.append(out["diagnostics"])
        probs_a = torch.sigmoid(out["action_final_candidate_logits"]).detach().cpu()
        probs_r = torch.sigmoid(out["reason_logits"]).detach().cpu()
        for i, fn in enumerate(batch["file_name"][:2]):
            failure_rows.append({
                "file_name": fn,
                "pred_action_candidate_score": probs_a[i].tolist(),
                "gt_action": action.detach().cpu()[i].tolist(),
                "pred_reason_top5": torch.topk(probs_r[i], k=min(5, probs_r.shape[1])).indices.tolist(),
                "gt_reason_indices": torch.where(reason.detach().cpu()[i] > 0.5)[0].tolist(),
                "primary_test_uses_bdd100k_gt": out["diagnostics"].get("primary_test_uses_bdd100k_gt", None),
            })
    cat = {k: torch.cat(v, 0) for k, v in tensors.items()}
    candidates = {
        "base": cat["action_base"],
        "evidence": cat["action_evidence"],
        "action_set": cat["action_set"],
        "candidate": cat["action_candidate"],
    }
    selected = select_guarded_action_branch(candidates, cat["labels_action"], threshold=args.threshold, margin=args.action_selector_margin)
    cat["action_guarded"] = selected["guarded_logits"]
    act_guard = multilabel_metrics_from_logits(cat["action_guarded"], cat["labels_action"], threshold=args.threshold, prefix="Act_guarded_")
    reason_base = multilabel_metrics_from_logits(cat["reason_base"], cat["labels_reason"], threshold=args.threshold, prefix="Exp_base_")
    reason_final = multilabel_metrics_from_logits(cat["reason_final"], cat["labels_reason"], threshold=args.threshold, prefix="Exp_")
    metrics = {**reason_base, **reason_final, **act_guard, "test_standard_joint": 0.5 * act_guard["Act_guarded_mF1"] + 0.5 * reason_final["Exp_mF1"], "epoch": epoch, "split": "test"}
    for branch, branch_metrics in selected["branch_metrics"].items():
        metrics.update(branch_metrics)
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    logits_dir = epoch_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in cat.items():
        torch.save(tensor, logits_dir / f"{name}_test.pt")
    write_json(logits_dir / "file_names_test.json", file_names)
    write_json(epoch_dir / "metrics_summary.json", metrics)
    write_json(epoch_dir / "metrics_raw_fixed.json", metrics)
    write_json(epoch_dir / "branch_metrics.json", selected["branch_metrics"])
    write_json(epoch_dir / "action_branch_selection.json", {k: v for k, v in selected.items() if k not in {"guarded_logits", "branch_metrics"}})
    write_json(epoch_dir / "action_evidence_stats.json", {"diagnostics": diagnostics})
    set_ids = []
    for d in diagnostics:
        if "selected_action_set_histogram" in d:
            set_ids.append(d["selected_action_set_histogram"])
    write_json(epoch_dir / "action_set_stats.json", {"selected_action_set_histogram": set_ids})
    write_json(epoch_dir / "per_label_reason_metrics.json", {"per_label_f1": metrics.get("Exp_per_label_f1"), "per_label_ap": metrics.get("Exp_per_label_ap")})
    tail = [5, 6, 9, 11, 12, 14]
    per_f1 = metrics.get("Exp_per_label_f1") or []
    per_ap = metrics.get("Exp_per_label_ap") or []
    write_json(epoch_dir / "tail_group_metrics.json", {"tail_indices": tail, "tail_f1": [per_f1[i] for i in tail if i < len(per_f1)], "tail_ap": [per_ap[i] for i in tail if i < len(per_ap)]})
    write_json(epoch_dir / "active_reason_stats.json", {"diagnostics": diagnostics})
    append_jsonl(epoch_dir / "failure_cases.jsonl", {"rows": failure_rows[:50]})
    append_jsonl(epoch_dir / "care_act_visuals_index.jsonl", {"rows": [{"file_name": x, "status": "diagnostic_schema", "epoch": epoch} for x in file_names[:16]]})
    return {"metrics": metrics, "tensors": cat, "file_names": file_names, "diagnostics": diagnostics, "selection": selected}


def train_one_epoch(args, backbone, model, loader, optimizer, device, epoch: int, out_dir: Path, action_residual_shutdown: bool = False) -> dict[str, Any]:
    model.train()
    losses = []
    diag_rows = []
    structured_totals = {"object_count": 0, "lane_count": 0, "drivable_count": 0, "attribute_count": 0, "has_structured": 0}
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        structured_totals["object_count"] += int(batch.get("object_count", torch.zeros(1)).sum().item())
        structured_totals["lane_count"] += int(batch.get("lane_count", torch.zeros(1)).sum().item())
        structured_totals["drivable_count"] += int(batch.get("drivable_count", torch.zeros(1)).sum().item())
        structured_totals["attribute_count"] += int(batch.get("attribute_count", torch.zeros(1)).sum().item())
        structured_totals["has_structured"] += int(batch.get("has_structured", torch.zeros(1)).sum().item())
        images = batch["image"].to(device, non_blocking=True)
        action = batch["action"].to(device)
        reason = batch["reason"].to(device)
        with torch.no_grad():
            tokens = extract_tokens(backbone, images, args.n_last_blocks)
        out = model(tokens, batch={"reason": reason}, structured=batch["bdd100k_structured"], epoch=epoch, action_residual_shutdown=action_residual_shutdown)
        loss, parts = care_act_training_loss(out, action, reason, args)
        (loss / args.gradient_accumulation_steps).backward()
        if step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(parts)
        diag_rows.append(out["diagnostics"])
        if step % args.log_every == 0:
            print(f"epoch={epoch} step={step}/{len(loader)} loss={parts['total_loss']:.5f} act={parts['action_loss']:.5f} exp={parts['reason_loss']:.5f} ae={parts['action_evidence_loss']:.5f} aset={parts['action_set_loss']:.5f}", flush=True)
        append_jsonl(out_dir / "loss_components.jsonl", {"epoch": epoch, "step": step, **parts})
    if len(loader) % args.gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    summary = _mean_dict(losses)
    summary["epoch"] = epoch
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    write_json(epoch_dir / "loss_components_summary.json", summary)
    write_json(epoch_dir / "expert_usage_stats.json", {"diagnostics": diag_rows})
    write_json(epoch_dir / "reason_update_stats.json", {"reason_residual_abs_max": [x.get("reason_residual_abs_max") for x in diag_rows]})
    write_json(epoch_dir / "action_safe_stats.json", {"action_total_delta_abs_max": [x.get("action_total_delta_abs_max") for x in diag_rows], "action_residual_shutdown": action_residual_shutdown})
    write_json(epoch_dir / "selected_vs_random_evidence_stats.json", {"selected_gt_random_drop_ratio": [x.get("selected_gt_random_drop_ratio") for x in diag_rows]})
    write_json(epoch_dir / "bdd100k_structured_stats.json", structured_totals)
    write_json(epoch_dir / "evidence_bag_stats.json", {"diagnostics": diag_rows})
    write_json(epoch_dir / "efficiency_stats.json", {"batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps})
    write_json(epoch_dir / "run_manifest_epoch.json", {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "test_only_evaluation": True})
    return {"loss": summary, "diagnostics": diag_rows}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_care_act_oia_v1.yaml")
    ap.add_argument("--data_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--bdd_oia_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--bdd100k_root", default="E:/sbw/BDD100K")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--loss_base_action", type=float, default=0.5)
    ap.add_argument("--loss_base_reason", type=float, default=0.2)
    ap.add_argument("--loss_action_visual", type=float, default=0.05)
    ap.add_argument("--loss_r2a_gt", type=float, default=0.1)
    ap.add_argument("--loss_action_agree", type=float, default=0.01)
    ap.add_argument("--loss_action_evidence", type=float, default=0.5)
    ap.add_argument("--loss_action_set", type=float, default=0.25)
    ap.add_argument("--loss_evidence_bag", type=float, default=0.1)
    ap.add_argument("--loss_reason_delta_reg", type=float, default=0.001)
    ap.add_argument("--loss_action_delta_reg", type=float, default=0.001)
    ap.add_argument("--action_selector_margin", type=float, default=0.006)
    ap.add_argument("--feature_cache_enabled", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--token_compression", default="none")
    ap.add_argument("--test_only_evaluation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--best_selection_split", default="test")
    ap.add_argument("--require_review_pass", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    _load_config(args)
    validate_args(args)
    if args.require_review_pass:
        require_review_pass(Path.cwd())
    out_dir = Path(args.output_dir or (Path(".background_runs") / f"care_act_oia_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    backbone, dim = build_backbone(args, device)
    model = CAREActOIAModel(dim=dim, action_dim=args.action_dim, reason_dim=args.reason_dim).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(args, "train", shuffle=True)
    test_loader = make_loader(args, "test", shuffle=False)
    manifest = {
        "repo": "FATE-OIA",
        "experiment": "care_act_oia_v1_direct_image",
        "command": " ".join(sys.argv),
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_resolved": vars(args),
        "feature_cache_enabled": False,
        "token_compression": "none",
        "test_only_evaluation": True,
        "best_selection_split": "test",
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "loss_divided_by_accumulation": True,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    best = -1.0
    action_residual_shutdown = False
    for epoch in range(args.epochs):
        print(f"=== CARE-ACT epoch {epoch}/{args.epochs - 1} train ===", flush=True)
        train_one_epoch(args, backbone, model, train_loader, optimizer, device, epoch, out_dir, action_residual_shutdown)
        print(f"=== CARE-ACT epoch {epoch} test ===", flush=True)
        test_stats = evaluate(args, backbone, model, test_loader, device, epoch, out_dir)
        metrics = test_stats["metrics"]
        append_jsonl(out_dir / "metrics_summary.jsonl", metrics)
        action_residual_shutdown = bool(test_stats["selection"]["shutdown_action_residual_next_epoch"])
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args), "metrics": metrics, "action_residual_shutdown_next_epoch": action_residual_shutdown}
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        if metrics["test_standard_joint"] > best:
            best = metrics["test_standard_joint"]
            torch.save(ckpt, out_dir / "checkpoint_best_test.pth")
            write_json(out_dir / "metrics_best_test.json", metrics)
        print(f"epoch={epoch} test_joint={metrics['test_standard_joint']:.6f} Act={metrics['Act_guarded_mF1']:.6f} Exp={metrics['Exp_mF1']:.6f} ExpAP={metrics['Exp_mAP']:.6f} branch={test_stats['selection']['selected_branch']}", flush=True)
    completion_payload = {
        "completed_epochs": args.epochs,
        "best_test_standard_joint": best,
        "output_dir": str(out_dir),
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "test_only_evaluation": args.test_only_evaluation,
        "best_selection_split": args.best_selection_split,
    }
    if is_full_goal_run(args):
        write_json(out_dir / "GOAL_COMPLETED_CARE_ACT_OIA_V1.json", completion_payload)
    else:
        completion_payload["reason"] = "smoke_or_partial_run_not_goal_completion"
        write_json(out_dir / "SMOKE_COMPLETED_CARE_ACT_OIA_V1.json", completion_payload)


if __name__ == "__main__":
    main()
