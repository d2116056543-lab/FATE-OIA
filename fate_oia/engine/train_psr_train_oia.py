from __future__ import annotations

import argparse
import json
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
from fate_oia.losses.psr_train_losses import psr_train_loss
from fate_oia.models.psr_train_oia_model import PSRTrainOIAFeatureModel
from fate_oia.utils.lr_scaling import compute_lr_scaling


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_config(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    for key, value in cfg.items():
        if hasattr(args, key) and getattr(args, key) == _DEFAULTS.get(key):
            setattr(args, key, value)
    return args


def _limit_dataset(ds, max_samples: int):
    if max_samples and max_samples > 0:
        return Subset(ds, list(range(min(len(ds), int(max_samples)))))
    return ds


def make_loader(args: argparse.Namespace, split: str, train: bool) -> DataLoader:
    ds = BDDOIAMultiTaskDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        split=split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=True,
        transform=build_transform(args.image_height, args.image_width, args.patch_size, args.preserve_aspect_ratio, return_meta=True),
    )
    ds = _limit_dataset(ds, args.max_train_samples if split == "train" else args.max_test_samples)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def labels_from_batch(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["action"].float(), batch["reason"].float()


def combined(action_logits: torch.Tensor, reason_logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([action_logits, reason_logits], dim=1)


def branch_metrics(outputs: dict[str, torch.Tensor], labels: torch.Tensor, action_dim: int, threshold: float) -> dict[str, dict[str, float]]:
    branches = {
        "final": combined(outputs["final_action_logits"], outputs["final_reason_logits"]),
        "action_specialist": combined(outputs["a_action_logits"], outputs["a_reason_logits"]),
        "explanation_specialist": combined(outputs["e_action_logits"], outputs["e_reason_logits"]),
        "calibration_specialist": combined(outputs["a_action_logits"], outputs["c_reason_logits"]),
    }
    return {
        name: evaluate_snna25(logits.cpu(), labels.cpu(), action_dim, threshold_mode="fixed", fixed_threshold=threshold)["metrics"]
        for name, logits in branches.items()
    }


def _joint(metrics: dict[str, float]) -> float:
    return 0.5 * float(metrics.get("Act_mF1", 0.0)) + 0.5 * float(metrics.get("Exp_mF1", 0.0))


def run_epoch(args, backbone, model, loader, optimizer, device, epoch: int, train: bool) -> dict[str, Any]:
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    accum = max(1, int(args.gradient_accumulation_steps))
    all_labels: list[torch.Tensor] = []
    out_lists: dict[str, list[torch.Tensor]] = {k: [] for k in [
        "final_action_logits", "final_reason_logits", "a_action_logits", "a_reason_logits",
        "e_action_logits", "e_reason_logits", "c_reason_logits",
    ]}
    file_names: list[str] = []
    losses: list[dict[str, Any]] = []
    total_loss = 0.0
    total_count = 0
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        action, reason = labels_from_batch(batch)
        action = action.to(device, non_blocking=True)
        reason = reason.to(device, non_blocking=True)
        with torch.no_grad():
            tokens = extract_tokens(backbone, images, args.n_last_blocks)
        outputs = model(tokens, epoch=epoch)
        loss, parts = psr_train_loss(outputs, action, reason, args)
        if train:
            (loss / float(accum)).backward()
            if ((step + 1) % accum == 0) or ((step + 1) == len(loader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        bs = int(images.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_count += bs
        labels = torch.cat([action.detach().cpu(), reason.detach().cpu()], dim=1)
        all_labels.append(labels)
        for key in out_lists:
            out_lists[key].append(outputs[key].detach().cpu())
        fn = batch.get("file_name", [])
        file_names.extend([str(x) for x in (fn if not isinstance(fn, str) else [fn])])
        row = {"epoch": epoch, "step": step, "train": train, **parts, "lr": float(optimizer.param_groups[0]["lr"]), "batch_size": bs}
        losses.append(row)
        if step % int(args.log_every) == 0:
            print(json.dumps({"event": "psr_train_batch", **row}, ensure_ascii=False), flush=True)
    labels_all = torch.cat(all_labels, 0) if all_labels else torch.empty(0, args.action_dim + args.reason_dim)
    outputs_all = {k: torch.cat(v, 0) if v else torch.empty(0, args.action_dim if "action" in k else args.reason_dim) for k, v in out_lists.items()}
    final_logits = combined(outputs_all["final_action_logits"], outputs_all["final_reason_logits"])
    metrics = evaluate_snna25(final_logits, labels_all, args.action_dim, threshold_mode="fixed", fixed_threshold=args.eval_threshold)["metrics"] if labels_all.numel() else {}
    branch = branch_metrics(outputs_all, labels_all, args.action_dim, args.eval_threshold) if labels_all.numel() else {}
    return {
        "loss": total_loss / max(1, total_count),
        "count": total_count,
        "metrics": metrics,
        "joint": _joint(metrics) if metrics else 0.0,
        "branch_metrics": branch,
        "labels": labels_all,
        "outputs": outputs_all,
        "file_names": file_names,
        "loss_components": losses,
    }


def save_split_outputs(out_dir: Path, split: str, stats: dict[str, Any], action_dim: int) -> None:
    tensors = stats["outputs"]
    torch.save(tensors["final_action_logits"], out_dir / f"logits_final_action_{split}.pt")
    torch.save(tensors["final_reason_logits"], out_dir / f"logits_final_reason_{split}.pt")
    torch.save(tensors["a_action_logits"], out_dir / f"logits_a_action_{split}.pt")
    torch.save(tensors["a_reason_logits"], out_dir / f"logits_a_reason_{split}.pt")
    torch.save(tensors["e_action_logits"], out_dir / f"logits_e_action_{split}.pt")
    torch.save(tensors["e_reason_logits"], out_dir / f"logits_e_reason_{split}.pt")
    torch.save(tensors["c_reason_logits"], out_dir / f"logits_c_reason_{split}.pt")
    labels = stats["labels"]
    torch.save(labels[:, :action_dim], out_dir / f"labels_action_{split}.pt")
    torch.save(labels[:, action_dim:], out_dir / f"labels_reason_{split}.pt")
    _write_json(out_dir / f"file_names_{split}.json", stats["file_names"])


def save_epoch_artifacts(out_dir: Path, epoch: int, train_stats: dict[str, Any], test_stats: dict[str, Any], manifest: dict[str, Any]) -> None:
    e_dir = out_dir / f"epoch_{epoch:03d}"
    e_dir.mkdir(parents=True, exist_ok=True)
    _write_json(e_dir / "metrics_summary.json", {"epoch": epoch, "train_loss": train_stats["loss"], "test": test_stats["metrics"], "branch_metrics": test_stats["branch_metrics"], "joint_test_score": test_stats["joint"]})
    for row in train_stats["loss_components"]:
        _append_jsonl(e_dir / "loss_components.jsonl", row)
    save_split_outputs(e_dir, "test", test_stats, int(manifest["config_resolved"]["action_dim"]))
    _write_json(e_dir / "run_manifest.json", manifest)


def build_manifest(args: argparse.Namespace, out_dir: Path, train_count: int, test_count: int) -> dict[str, Any]:
    return {
        "repo_name": "FATE-OIA",
        "method": "PSR-Train OIA V1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "data_root": args.data_root,
        "raw_root": args.raw_root,
        "pretrained_weights": args.pretrained_weights,
        "initialization_policy": "DINO_pretrained_only_no_old_checkpoint_resume",
        "resume_policy": "same_output_dir_psr_train_checkpoint_only",
        "resume_psr_train_checkpoint": args.resume_psr_train_checkpoint,
        "uses_old_logits_for_training": False,
        "uses_feature_cache": False,
        "eval_splits": ["test"],
        "best_selection_split": "test",
        "best_selection_metric": "joint_test_score",
        "train_split_count": int(train_count),
        "test_split_count": int(test_count),
        "output_dir": str(out_dir),
        "config_resolved": vars(args),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train end-to-end PSR-Train OIA V1 from DINO pretrained image tokens.")
    ap.add_argument("--config", default="")
    ap.add_argument("--data_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--raw_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--auto_scale_lr", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reference_effective_batch", type=int, default=32)
    ap.add_argument("--base_head_lr_at_reference_batch", type=float, default=3e-4)
    ap.add_argument("--max_head_lr", type=float, default=5e-4)
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--specialist_warmup_epochs", type=int, default=4)
    ap.add_argument("--router_warmup_epochs", type=int, default=10)
    ap.add_argument("--action_delta_cap", type=float, default=0.04)
    ap.add_argument("--loss_final_action", type=float, default=1.0)
    ap.add_argument("--loss_final_reason", type=float, default=1.0)
    ap.add_argument("--loss_a_action", type=float, default=0.4)
    ap.add_argument("--loss_e_reason", type=float, default=0.4)
    ap.add_argument("--loss_a_reason", type=float, default=0.05)
    ap.add_argument("--loss_e_action", type=float, default=0.01)
    ap.add_argument("--loss_calibration_reason", type=float, default=0.05)
    ap.add_argument("--loss_pareto", type=float, default=0.2)
    ap.add_argument("--loss_gradient_budget", type=float, default=0.001)
    ap.add_argument("--pareto_margin_action", type=float, default=0.005)
    ap.add_argument("--pareto_margin_reason", type=float, default=0.005)
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=60)
    ap.add_argument("--resume_psr_train_checkpoint", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    global _DEFAULTS
    _DEFAULTS = vars(ap.parse_args(["--output_dir", "__dummy__"]))
    _apply_config(args, _load_config(args.config))
    args.num_gpus = 1
    lr_info = compute_lr_scaling(
        per_gpu_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        reference_effective_batch=args.reference_effective_batch,
        base_lr_at_reference_batch=args.base_head_lr_at_reference_batch,
        num_gpus=1,
        auto_scale_lr=args.auto_scale_lr,
        current_lr=args.lr,
        max_lr=args.max_head_lr,
    )
    args.effective_batch_size = lr_info.effective_batch_size
    if args.auto_scale_lr:
        args.lr = lr_info.lr_actual
    if not Path(args.pretrained_weights).exists():
        raise FileNotFoundError(f"pretrained_weights does not exist: {args.pretrained_weights}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "args.json", vars(args))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    backbone, dim = build_backbone(args, device)
    model = PSRTrainOIAFeatureModel(dim=dim, action_dim=args.action_dim, reason_dim=args.reason_dim, action_delta_cap=args.action_delta_cap, specialist_warmup_epochs=args.specialist_warmup_epochs, router_warmup_epochs=args.router_warmup_epochs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_score = -1.0
    if args.resume_psr_train_checkpoint:
        resume_path = Path(args.resume_psr_train_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_psr_train_checkpoint does not exist: {resume_path}")
        if resume_path.resolve().parent != out_dir.resolve():
            raise ValueError("resume_psr_train_checkpoint must be inside the same output_dir; old RunC/CARE checkpoints are not allowed")
        resume = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume["model"], strict=True)
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume.get("epoch", -1)) + 1
        best_score = float(resume.get("best_test_score", -1.0))
        print(json.dumps({
            "event": "psr_train_resume",
            "resume_psr_train_checkpoint": str(resume_path),
            "start_epoch": start_epoch,
            "best_test_score": best_score,
            "same_output_dir_only": True,
        }, ensure_ascii=False), flush=True)
    train_loader = make_loader(args, "train", True)
    test_loader = make_loader(args, "test", False)
    manifest = build_manifest(args, out_dir, len(train_loader.dataset), len(test_loader.dataset))
    _write_json(out_dir / "run_manifest.json", manifest)
    for epoch in range(start_epoch, args.epochs):
        train_stats = run_epoch(args, backbone, model, train_loader, optimizer, device, epoch, True)
        test_stats = run_epoch(args, backbone, model, test_loader, optimizer, device, epoch, False)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "test_loss": test_stats["loss"],
            "joint_test_score": test_stats["joint"],
            "test_metrics": test_stats["metrics"],
            "branch_metrics": test_stats["branch_metrics"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "best_selection_split": "test",
        }
        _append_jsonl(out_dir / "metrics_summary.jsonl", row)
        _write_json(out_dir / "metrics_latest.json", row)
        save_epoch_artifacts(out_dir, epoch, train_stats, test_stats, manifest)
        save_split_outputs(out_dir, "test", test_stats, args.action_dim)
        ckpt = {
            "epoch": epoch,
            "method": "PSR-Train OIA V1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "best_test_score": max(best_score, test_stats["joint"]),
        }
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        if test_stats["joint"] >= best_score:
            best_score = test_stats["joint"]
            torch.save(ckpt, out_dir / "checkpoint_best_test.pth")
            torch.save(ckpt, out_dir / "checkpoint_best.pth")
            _write_json(out_dir / "metrics_best_test.json", row)
        print(json.dumps({"event": "psr_train_epoch", **row}, ensure_ascii=False), flush=True)
    _write_json(out_dir / "TRAIN_COMPLETED_PSR_TRAIN_OIA_V1.json", {"best_test_score": best_score, "output_dir": str(out_dir), "test_only": True})


if __name__ == "__main__":
    main()
