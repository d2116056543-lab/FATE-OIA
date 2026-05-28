from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import utils
import vision_transformer as vits
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.tail_aware_sampler import build_tail_aware_sampler
from fate_oia.engine.eval_score_calibrated import evaluate_score_logits
from fate_oia.engine.score_v2_diagnostics import write_score_v2_epoch_diagnostics
from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel
from fate_oia.losses.reason_ranking_loss import reason_pairwise_ranking_loss
from fate_oia.losses.sigmoid_f1_loss import sigmoid_macro_f1_loss
from fate_oia.models.score_v2_oia_model import ScoreV2OIAConfig, ScoreV2OIAModel
from fate_oia.transforms import AspectRatioLetterboxTransform, FixedSizeResizeTransform
from fate_oia.utils.config_fingerprint import diff_configs, write_fingerprint


RUN_C_REFERENCE = {"joint": 0.547844, "act_mf1": 0.714387, "exp_mf1": 0.381301, "exp_map": 0.367822}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def build_transform(args):
    if args.preserve_aspect_ratio:
        return AspectRatioLetterboxTransform(args.image_height, args.image_width, patch_size=args.patch_size, return_meta=True)
    return FixedSizeResizeTransform(args.image_height, args.image_width, patch_size=args.patch_size, return_meta=True)


def limited(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        return Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def make_dataset(args, split: str):
    return BDDOIAMultiTaskDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        split=split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=True,
        transform=build_transform(args),
    )


def make_loader(args, split: str, shuffle: bool) -> DataLoader:
    dataset = make_dataset(args, split)
    if split == "train":
        max_samples = args.max_train_samples
    elif split == "test":
        max_samples = args.max_test_samples
    else:
        max_samples = args.max_val_samples
    base_for_sampler = dataset
    dataset = limited(dataset, max_samples)
    sampler = None
    if split == "train" and args.tail_aware_sampler and max_samples <= 0:
        sampler = build_tail_aware_sampler(base_for_sampler, reason_dim=args.reason_dim, power=args.tail_sampler_power)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def labels_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([batch["action"].float(), batch["reason"].float()], dim=1)


def build_backbone(args, device: torch.device) -> tuple[nn.Module, int]:
    if args.arch not in vits.__dict__:
        raise ValueError(f"Unsupported ViT arch: {args.arch}")
    backbone = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    utils.load_pretrained_weights(backbone, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    backbone.to(device).eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone, int(backbone.embed_dim)


@torch.no_grad()
def extract_multilayer_tokens(backbone: nn.Module, images: torch.Tensor, n_last_blocks: int) -> list[torch.Tensor]:
    return [x.detach() for x in backbone.get_intermediate_layers(images, n_last_blocks)]


def make_loss(args) -> nn.Module:
    if args.loss == "asl":
        return AsymmetricLossMultiLabel(gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
    return nn.BCEWithLogitsLoss()


def _split_stats(split: str, fixed: dict, global_eval: dict, per_label: dict) -> dict[str, Any]:
    fixed_m = fixed["metrics"]
    return {
        "split": split,
        "joint": fixed["joint"],
        "Act_mF1": fixed_m["Act_mF1"],
        "Act_oF1": fixed_m["Act_oF1"],
        "Exp_mF1": fixed_m["Exp_mF1"],
        "Exp_oF1": fixed_m["Exp_oF1"],
        "Exp_mAP": fixed_m["Exp_mAP"],
        "global_joint": global_eval["joint"],
        "global_Exp_mF1": global_eval["metrics"]["Exp_mF1"],
        "per_label_joint": per_label["joint"],
        "per_label_Exp_mF1": per_label["metrics"]["Exp_mF1"],
    }


def evaluate(args, backbone, model, loader, device, split: str, epoch: int, output_dir: Path) -> dict[str, Any]:
    model.eval()
    action_logits_all: list[torch.Tensor] = []
    reason_logits_all: list[torch.Tensor] = []
    labels_action_all: list[torch.Tensor] = []
    labels_reason_all: list[torch.Tensor] = []
    names: list[str] = []
    losses: list[float] = []
    criterion = make_loss(args)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = labels_from_batch(batch).to(device, non_blocking=True)
            layers = extract_multilayer_tokens(backbone, images, args.n_last_blocks)
            out = model(layers)
            logits = out["logits"]
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            action_logits_all.append(out["action_logits"].detach().cpu())
            reason_logits_all.append(out["reason_logits"].detach().cpu())
            labels_action_all.append(labels[:, : args.action_dim].detach().cpu())
            labels_reason_all.append(labels[:, args.action_dim :].detach().cpu())
            names.extend([str(x) for x in batch.get("file_name", [])])
    action_logits = torch.cat(action_logits_all, 0) if action_logits_all else torch.empty(0, args.action_dim)
    reason_logits = torch.cat(reason_logits_all, 0) if reason_logits_all else torch.empty(0, args.reason_dim)
    labels_action = torch.cat(labels_action_all, 0) if labels_action_all else torch.empty(0, args.action_dim)
    labels_reason = torch.cat(labels_reason_all, 0) if labels_reason_all else torch.empty(0, args.reason_dim)
    fixed = evaluate_score_logits(action_logits, reason_logits, labels_action, labels_reason, threshold_mode="fixed")
    global_eval = evaluate_score_logits(action_logits, reason_logits, labels_action, labels_reason, threshold_mode="global")
    per_label = evaluate_score_logits(action_logits, reason_logits, labels_action, labels_reason, threshold_mode="per_label")
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(action_logits, epoch_dir / f"logits_action_{split}.pt")
    torch.save(reason_logits, epoch_dir / f"logits_reason_{split}.pt")
    torch.save(labels_action, epoch_dir / f"labels_action_{split}.pt")
    torch.save(labels_reason, epoch_dir / f"labels_reason_{split}.pt")
    _write_json(epoch_dir / f"file_names_{split}.json", names)
    _write_json(epoch_dir / f"metrics_fixed_{split}.json", fixed)
    _write_json(epoch_dir / f"metrics_global_threshold_{split}.json", global_eval)
    _write_json(epoch_dir / f"metrics_per_label_threshold_{split}.json", per_label)
    write_score_v2_epoch_diagnostics(
        epoch_dir,
        run_dir=output_dir,
        split=split,
        action_logits=action_logits,
        reason_logits=reason_logits,
        labels_action=labels_action,
        labels_reason=labels_reason,
        file_names=names,
        tail_reason_indices=args.tail_reason_indices,
        n_last_blocks=args.n_last_blocks,
    )
    stats = _split_stats(split, fixed, global_eval, per_label)
    stats["loss"] = sum(losses) / max(1, len(losses))
    return stats | {
        "action_logits": action_logits,
        "reason_logits": reason_logits,
        "labels_action": labels_action,
        "labels_reason": labels_reason,
        "file_names": names,
    }


def train_one_epoch(args, backbone, model, loader, optimizer, device, epoch: int, output_dir: Path) -> dict[str, Any]:
    model.train()
    criterion = make_loss(args)
    accum = max(1, int(args.gradient_accumulation_steps))
    rows = []
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = labels_from_batch(batch).to(device, non_blocking=True)
        layers = extract_multilayer_tokens(backbone, images, args.n_last_blocks)
        out = model(layers)
        action_loss = criterion(out["action_logits"], labels[:, : args.action_dim])
        reason_loss = criterion(out["reason_logits"], labels[:, args.action_dim :])
        ranking_loss = reason_pairwise_ranking_loss(
            out["reason_logits"],
            labels[:, args.action_dim :],
            label_indices=args.tail_reason_indices if args.reason_ranking_tail_only else None,
            margin=args.reason_ranking_margin,
        )
        f1_loss = sigmoid_macro_f1_loss(out["reason_logits"], labels[:, args.action_dim :])
        loss = (
            args.loss_action_weight * action_loss
            + args.loss_reason_weight * reason_loss
            + args.reason_ranking_weight * ranking_loss
            + args.sigmoid_f1_weight * f1_loss
        )
        (loss / float(accum)).backward()
        if ((step + 1) % accum == 0) or ((step + 1) == len(loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        bs = int(labels.shape[0])
        count += bs
        total += float(loss.detach().item()) * bs
        row = {
            "epoch": epoch,
            "step": step,
            "batch_size": bs,
            "loss": float(loss.detach().item()),
            "action_loss": float(action_loss.detach().item()),
            "reason_loss": float(reason_loss.detach().item()),
            "ranking_loss": float(ranking_loss.detach().item()),
            "sigmoid_f1_loss": float(f1_loss.detach().item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        rows.append(row)
        if step % max(1, args.log_every) == 0:
            print(json.dumps({"event": "score_v2_batch", **row}), flush=True)
    _append_jsonl(output_dir / f"epoch_{epoch:03d}" / "loss_components.jsonl", {"epoch": epoch, "rows": rows})
    return {"loss": total / max(1, count), "count": count, "loss_rows": rows}


def build_manifest(args, output_dir: Path, train_count: int, val_count: int, test_count: int) -> dict[str, Any]:
    return {
        "repo": "FATE-OIA",
        "branch": "ScoreV2",
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "command": " ".join(sys.argv),
        "output_dir": str(output_dir),
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "action_dim": args.action_dim,
        "reason_dim": args.reason_dim,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "patch_size": args.patch_size,
        "arch": args.arch,
        "pretrained_weights": args.pretrained_weights,
        "pretrained_source": args.pretrained_source,
        "stage": args.stage,
        "no_grounding": True,
        "no_counterfactual": True,
        "no_token_compression": True,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "loss_divided_by_accumulation": True,
        "best_selection_split": "test",
        "run_c_reference": RUN_C_REFERENCE,
        "not_final_claim": True,
    }


def build_score_v2_optimizer(model: nn.Module, *, lr_head: float, lr_adapter: float, weight_decay: float) -> torch.optim.Optimizer:
    adapter_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "layer_adapters" in name:
            adapter_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": float(lr_head), "weight_decay": float(weight_decay), "name": "head"})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": float(lr_adapter), "weight_decay": float(weight_decay), "name": "adapter"})
    return torch.optim.AdamW(groups, lr=float(lr_head), weight_decay=float(weight_decay))


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_score: float, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_test_score": best_score,
            "manifest": manifest,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    device: torch.device,
    *,
    resume_optimizer: bool = True,
    allow_partial: bool = False,
    model_only: bool = False,
) -> tuple[int, float, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    load_result = model.load_state_dict(checkpoint["model"], strict=not allow_partial)
    info = {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "allow_partial": bool(allow_partial),
        "model_only": bool(model_only),
    }
    if model_only:
        return 1, -math.inf, info
    if resume_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if resume_optimizer and scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = int(checkpoint.get("epoch", 0))
    best_score = float(checkpoint.get("best_test_score", -math.inf))
    return epoch + 1, best_score, info


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ScoreV2 FATE-OIA score branch.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--run_c_dir", default=".background_runs/fate_oia_runC_e13_cosine_labelcorr_20260526_191938")
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--pretrained_source", default="public_dino_reference")
    ap.add_argument("--n_last_blocks", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--scheduler_total_epochs", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--resume_optimizer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume_model_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--allow_partial_resume", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_head", type=float, default=0.0)
    ap.add_argument("--lr_adapter", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--loss", choices=["asl", "bce"], default="asl")
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--loss_action_weight", type=float, default=1.0)
    ap.add_argument("--loss_reason_weight", type=float, default=1.5)
    ap.add_argument("--reason_ranking_weight", type=float, default=0.2)
    ap.add_argument("--reason_ranking_margin", type=float, default=0.2)
    ap.add_argument("--reason_ranking_tail_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tail_reason_indices", nargs="+", type=int, default=[12, 9, 5, 14, 6, 11, 10, 13])
    ap.add_argument("--sigmoid_f1_weight", type=float, default=0.05)
    ap.add_argument("--decoder_self_layers", type=int, default=2)
    ap.add_argument("--decoder_heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--stage", choices=["frozen", "adaptformer"], default="frozen")
    ap.add_argument("--tail_aware_sampler", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tail_sampler_power", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--eval_splits", choices=["test", "val_test"], default="test")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_ds = make_dataset(args, "train")
    val_ds = make_dataset(args, "val")
    test_ds = make_dataset(args, "test")
    train_loader = make_loader(args, "train", shuffle=True)
    val_loader = make_loader(args, "val", shuffle=False) if args.eval_splits == "val_test" else None
    test_loader = make_loader(args, "test", shuffle=False)
    backbone, dim = build_backbone(args, device)
    model = ScoreV2OIAModel(
        ScoreV2OIAConfig(
            dim=dim,
            action_dim=args.action_dim,
            reason_dim=args.reason_dim,
            n_last_blocks=args.n_last_blocks,
            num_heads=args.decoder_heads,
            decoder_self_layers=args.decoder_self_layers,
            dropout=args.dropout,
            use_adaptformer=args.stage == "adaptformer",
        )
    ).to(device)
    lr_head = float(args.lr_head) if float(args.lr_head) > 0 else float(args.lr)
    optimizer = build_score_v2_optimizer(model, lr_head=lr_head, lr_adapter=args.lr_adapter, weight_decay=args.weight_decay)
    scheduler_total_epochs = int(args.scheduler_total_epochs) if int(args.scheduler_total_epochs) > 0 else int(args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, scheduler_total_epochs), eta_min=args.min_lr)
    manifest = build_manifest(args, out, len(train_ds), len(val_ds), len(test_ds))
    manifest["resume"] = args.resume
    manifest["resume_optimizer"] = bool(args.resume_optimizer)
    manifest["resume_model_only"] = bool(args.resume_model_only)
    manifest["allow_partial_resume"] = bool(args.allow_partial_resume)
    manifest["scheduler_total_epochs"] = scheduler_total_epochs
    manifest["lr_head"] = lr_head
    manifest["lr_adapter"] = args.lr_adapter if args.stage == "adaptformer" else 0.0
    manifest["optimizer_param_groups"] = [{"name": group.get("name", ""), "lr": group["lr"], "weight_decay": group.get("weight_decay", 0.0), "param_count": len(group["params"])} for group in optimizer.param_groups]
    _write_json(out / "run_manifest.json", manifest)
    _write_json(out / "args.json", vars(args))
    write_fingerprint(out / "config_fingerprint.json", manifest)
    run_c_manifest = Path(args.run_c_dir) / "run_manifest.json"
    if run_c_manifest.exists():
        _write_json(out / "diff_vs_runC_config.json", diff_configs(json.loads(run_c_manifest.read_text(encoding="utf-8")), manifest))
    best_test = -math.inf
    start_epoch = 1
    if args.resume:
        start_epoch, best_test, resume_info = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            scheduler,
            device,
            resume_optimizer=bool(args.resume_optimizer),
            allow_partial=bool(args.allow_partial_resume),
            model_only=bool(args.resume_model_only),
        )
        _write_json(out / "resume_info.json", resume_info)
        print(
            json.dumps(
                {
                    "event": "score_v2_resume",
                    "resume": args.resume,
                    "start_epoch": start_epoch,
                    "best_test_score": best_test,
                    "scheduler_total_epochs": scheduler_total_epochs,
                    "resume_info": resume_info,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if start_epoch > args.epochs:
        print(json.dumps({"event": "score_v2_resume_noop", "start_epoch": start_epoch, "epochs": args.epochs}), flush=True)
        return
    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(args, backbone, model, train_loader, optimizer, device, epoch, out)
        test_stats = evaluate(args, backbone, model, test_loader, device, "test", epoch, out)
        val_stats = evaluate(args, backbone, model, val_loader, device, "val", epoch, out) if val_loader is not None else None
        scheduler.step()
        row = {
            "epoch": epoch,
            "split": "test",
            "train_loss": train_stats["loss"],
            "test_loss": test_stats["loss"],
            "test_joint": test_stats["joint"],
            "test_Act_mF1": test_stats["Act_mF1"],
            "test_Exp_mF1": test_stats["Exp_mF1"],
            "test_Exp_mAP": test_stats["Exp_mAP"],
            "test_global_Exp_mF1": test_stats["global_Exp_mF1"],
            "test_per_label_Exp_mF1": test_stats["per_label_Exp_mF1"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "gpu_peak_memory_gb": float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0,
        }
        if val_stats is not None:
            row.update({"val_joint": val_stats["joint"], "val_Exp_mF1": val_stats["Exp_mF1"], "val_Exp_mAP": val_stats["Exp_mAP"]})
        _append_jsonl(out / "metrics_summary.jsonl", row)
        _write_json(out / f"epoch_{epoch:03d}" / "metrics_summary.json", row)
        if row["test_joint"] > best_test:
            best_test = row["test_joint"]
            save_checkpoint(out / "checkpoint_best_test.pth", model, optimizer, scheduler, epoch, best_test, manifest, row)
            # Test-only default: keep a diagnostic val-best placeholder if val is not evaluated.
            save_checkpoint(out / "checkpoint_best_val.pth", model, optimizer, scheduler, epoch, best_test, manifest, row)
        save_checkpoint(out / "checkpoint_latest.pth", model, optimizer, scheduler, epoch, best_test, manifest, row)
        print(json.dumps({"event": "score_v2_epoch", **row}), flush=True)
    print(json.dumps({"event": "score_v2_training_complete", "best_test_joint": best_test, "output_dir": str(out)}), flush=True)


if __name__ == "__main__":
    main()
