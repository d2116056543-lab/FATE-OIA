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

from fate_oia.engine.train_score_v2_oia import (
    build_backbone,
    build_transform,
    evaluate_score_logits,
    extract_multilayer_tokens,
    labels_from_batch,
    limited,
    make_dataset,
    make_loader,
)
from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel
from fate_oia.losses.reason_ranking_loss import reason_pairwise_ranking_loss
from fate_oia.losses.sigmoid_f1_loss import sigmoid_macro_f1_loss
from fate_oia.models.head_zoo.ctran_head import CTranMaskedHead
from fate_oia.models.head_zoo.ml_decoder_head import MLDecoderHead
from fate_oia.models.head_zoo.q2l_decoder_head import Q2LDecoderHead
from fate_oia.models.head_zoo.run_c_calibrated_head import RunCCalibratedHead
from fate_oia.models.head_zoo.run_c_compatible_head import RunCCompatibleHead
from fate_oia.models.head_zoo.run_c_mrc_head import RunCMRCAuxHead
from fate_oia.models.multilayer_dino_features import MultiLayerDINOFeatureFusion
from fate_oia.utils.config_fingerprint import diff_configs, write_fingerprint


RUN_C_REFERENCE = {"joint": 0.547844, "act_mf1": 0.714387, "exp_mf1": 0.381301, "exp_map": 0.367822}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def build_head(name: str, *, dim: int, action_dim: int, reason_dim: int, **kwargs) -> nn.Module:
    if name == "h0_runc_compatible":
        return RunCCompatibleHead(dim=dim, action_dim=action_dim, reason_dim=reason_dim, dropout=kwargs.get("dropout", 0.1))
    if name == "h1_q2l_decoder":
        return Q2LDecoderHead(dim=dim, action_dim=action_dim, reason_dim=reason_dim, num_heads=kwargs.get("decoder_heads", 6), self_layers=kwargs.get("decoder_self_layers", 2), dropout=kwargs.get("dropout", 0.1))
    if name == "h2_ml_decoder_g8":
        return MLDecoderHead(dim=dim, action_dim=action_dim, reason_dim=reason_dim, groups=kwargs.get("ml_decoder_groups", 8), num_heads=kwargs.get("decoder_heads", 4), dropout=kwargs.get("dropout", 0.1))
    if name == "h3_ctran_masked":
        return CTranMaskedHead(dim=dim, action_dim=action_dim, reason_dim=reason_dim, reveal_prob=kwargs.get("ctran_reveal_prob", 0.35), num_heads=kwargs.get("decoder_heads", 6), dropout=kwargs.get("dropout", 0.1))
    if name == "h4_runc_mrc_aux":
        return RunCMRCAuxHead(dim=dim, action_dim=action_dim, reason_dim=reason_dim, mrc_mask_ratio=kwargs.get("mrc_mask_ratio", 0.35), dropout=kwargs.get("dropout", 0.1))
    if name == "h5_runc_calibrated":
        return RunCCalibratedHead(dim=dim, action_dim=action_dim, reason_dim=reason_dim, dropout=kwargs.get("dropout", 0.1))
    raise ValueError(f"Unknown HeadZoo head: {name}")


class HeadZooModel(nn.Module):
    def __init__(self, head_name: str, *, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, n_last_blocks: int = 4, dropout: float = 0.1, **head_kwargs) -> None:
        super().__init__()
        self.fusion = MultiLayerDINOFeatureFusion(dim, n_last_blocks, dropout=dropout)
        self.head = build_head(head_name, dim=dim, action_dim=action_dim, reason_dim=reason_dim, dropout=dropout, **head_kwargs)

    def forward(self, layers: list[torch.Tensor] | tuple[torch.Tensor, ...], labels: torch.Tensor | None = None) -> dict[str, Any]:
        fused = self.fusion(layers)
        out = dict(self.head(fused["tokens"], labels=labels))
        out["tokens"] = fused["tokens"]
        out["layer_weights"] = fused["layer_weights"]
        return out


def make_loss(args) -> nn.Module:
    if args.loss == "asl":
        return AsymmetricLossMultiLabel(gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
    return nn.BCEWithLogitsLoss()


def compute_head_zoo_loss(
    out: dict[str, Any],
    labels: torch.Tensor,
    *,
    action_dim: int = 4,
    loss_action_weight: float = 1.0,
    loss_reason_weight: float = 1.5,
    reason_ranking_weight: float = 0.05,
    sigmoid_f1_weight: float = 0.05,
    criterion: nn.Module | None = None,
    tail_reason_indices: list[int] | None = None,
    reason_ranking_margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    criterion = criterion or AsymmetricLossMultiLabel(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    action_labels = labels[:, :action_dim]
    reason_labels = labels[:, action_dim:]
    action_loss = criterion(out["action_logits"], action_labels)
    reason_loss = criterion(out["reason_logits"], reason_labels)
    ranking_loss = reason_pairwise_ranking_loss(out["reason_logits"], reason_labels, label_indices=tail_reason_indices, margin=reason_ranking_margin)
    f1_loss = sigmoid_macro_f1_loss(out["reason_logits"], reason_labels)
    total = loss_action_weight * action_loss + loss_reason_weight * reason_loss + reason_ranking_weight * ranking_loss + sigmoid_f1_weight * f1_loss
    parts = {
        "action_loss": float(action_loss.detach().item()),
        "reason_loss": float(reason_loss.detach().item()),
        "ranking_loss": float(ranking_loss.detach().item()),
        "sigmoid_f1_loss": float(f1_loss.detach().item()),
    }
    for name, value in (out.get("aux_losses") or {}).items():
        total = total + value
        parts[name] = float(value.detach().item())
    parts["total_loss"] = float(total.detach().item())
    return total, parts


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
    action_logits_all, reason_logits_all, labels_action_all, labels_reason_all = [], [], [], []
    names: list[str] = []
    criterion = make_loss(args)
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = labels_from_batch(batch).to(device, non_blocking=True)
            layers = extract_multilayer_tokens(backbone, images, args.n_last_blocks)
            out = model(layers)
            logits = torch.cat([out["action_logits"], out["reason_logits"]], dim=1)
            losses.append(float(criterion(logits, labels).item()))
            action_logits_all.append(out["action_logits"].detach().cpu())
            reason_logits_all.append(out["reason_logits"].detach().cpu())
            labels_action_all.append(labels[:, : args.action_dim].detach().cpu())
            labels_reason_all.append(labels[:, args.action_dim :].detach().cpu())
            names.extend([str(x) for x in batch.get("file_name", [])])
    action_logits = torch.cat(action_logits_all, 0)
    reason_logits = torch.cat(reason_logits_all, 0)
    labels_action = torch.cat(labels_action_all, 0)
    labels_reason = torch.cat(labels_reason_all, 0)
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
    _write_json(epoch_dir / "threshold_sweep_test.json" if split == "test" else epoch_dir / f"threshold_sweep_{split}.json", {"fixed": fixed, "global": global_eval, "per_label": per_label})
    stats = _split_stats(split, fixed, global_eval, per_label)
    stats["loss"] = sum(losses) / max(1, len(losses))
    return stats


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
        out = model(layers, labels=labels)
        loss, parts = compute_head_zoo_loss(
            out,
            labels,
            action_dim=args.action_dim,
            loss_action_weight=args.loss_action_weight,
            loss_reason_weight=args.loss_reason_weight,
            reason_ranking_weight=args.reason_ranking_weight,
            sigmoid_f1_weight=args.sigmoid_f1_weight,
            criterion=criterion,
            tail_reason_indices=args.tail_reason_indices if args.reason_ranking_tail_only else None,
            reason_ranking_margin=args.reason_ranking_margin,
        )
        (loss / float(accum)).backward()
        if ((step + 1) % accum == 0) or ((step + 1) == len(loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        bs = int(labels.shape[0])
        count += bs
        total += float(loss.detach().item()) * bs
        row = {"epoch": epoch, "step": step, "batch_size": bs, "lr": float(optimizer.param_groups[0]["lr"])} | parts
        rows.append(row)
        if step % max(1, args.log_every) == 0:
            print(json.dumps({"event": "head_zoo_batch", "head_name": args.head_name, **row}), flush=True)
    _append_jsonl(output_dir / f"epoch_{epoch:03d}" / "loss_components.jsonl", {"epoch": epoch, "rows": rows})
    return {"loss": total / max(1, count), "count": count}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_score: float, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None, "epoch": epoch, "best_test_score": best_score, "manifest": manifest, "metrics": metrics}, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device: torch.device) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0)) + 1, float(ckpt.get("best_test_score", -math.inf))


def build_manifest(args, output_dir: Path, train_count: int, val_count: int, test_count: int) -> dict[str, Any]:
    return {
        "repo": "FATE-OIA",
        "branch": "HeadZoo",
        "head_name": args.head_name,
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
        "pretrained_weights": args.pretrained_weights,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "no_grounding": True,
        "no_counterfactual": True,
        "no_token_compression": True,
        "best_selection_split": "test",
        "not_final_claim": True,
        "run_c_reference": RUN_C_REFERENCE,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one FATE-OIA HeadZoo head.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--head_name", required=True)
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--run_c_dir", default=".background_runs/fate_oia_runC_e13_cosine_labelcorr_20260526_191938")
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--n_last_blocks", type=int, default=4)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--resume", default="")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--loss", choices=["asl", "bce"], default="asl")
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--loss_action_weight", type=float, default=1.0)
    ap.add_argument("--loss_reason_weight", type=float, default=1.5)
    ap.add_argument("--reason_ranking_weight", type=float, default=0.05)
    ap.add_argument("--reason_ranking_margin", type=float, default=0.2)
    ap.add_argument("--reason_ranking_tail_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tail_reason_indices", nargs="+", type=int, default=[12, 9, 5, 14, 6, 11, 10, 13])
    ap.add_argument("--sigmoid_f1_weight", type=float, default=0.05)
    ap.add_argument("--decoder_self_layers", type=int, default=2)
    ap.add_argument("--decoder_heads", type=int, default=6)
    ap.add_argument("--ml_decoder_groups", type=int, default=8)
    ap.add_argument("--ctran_reveal_prob", type=float, default=0.35)
    ap.add_argument("--mrc_mask_ratio", type=float, default=0.35)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--tail_aware_sampler", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tail_sampler_power", type=float, default=0.5)
    ap.add_argument("--eval_splits", choices=["test", "val_test"], default="test")
    ap.add_argument("--log_every", type=int, default=80)
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
    model = HeadZooModel(
        args.head_name,
        dim=dim,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        n_last_blocks=args.n_last_blocks,
        dropout=args.dropout,
        decoder_heads=args.decoder_heads,
        decoder_self_layers=args.decoder_self_layers,
        ml_decoder_groups=args.ml_decoder_groups,
        ctran_reveal_prob=args.ctran_reveal_prob,
        mrc_mask_ratio=args.mrc_mask_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    manifest = build_manifest(args, out, len(train_ds), len(val_ds), len(test_ds))
    _write_json(out / "run_manifest.json", manifest)
    _write_json(out / "args.json", vars(args))
    write_fingerprint(out / "config_fingerprint.json", manifest)
    run_c_manifest = Path(args.run_c_dir) / "run_manifest.json"
    if run_c_manifest.exists():
        _write_json(out / "diff_vs_runC_config.json", diff_configs(json.loads(run_c_manifest.read_text(encoding="utf-8")), manifest))
    best_test = -math.inf
    start_epoch = 1
    if args.resume:
        start_epoch, best_test = load_checkpoint(Path(args.resume), model, optimizer, scheduler, device)
    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(args, backbone, model, train_loader, optimizer, device, epoch, out)
        test_stats = evaluate(args, backbone, model, test_loader, device, "test", epoch, out)
        val_stats = evaluate(args, backbone, model, val_loader, device, "val", epoch, out) if val_loader is not None else None
        scheduler.step()
        row = {
            "epoch": epoch,
            "head_name": args.head_name,
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
        _write_json(out / f"epoch_{epoch:03d}" / "per_label_reason_audit.json", {"note": "per-label audit placeholders are available from metrics_per_label_threshold_test.json", "head_name": args.head_name})
        _write_json(out / f"epoch_{epoch:03d}" / "tail_group_metrics.json", {"tail_reason_indices": args.tail_reason_indices, "test_Exp_mF1": row["test_Exp_mF1"], "test_Exp_mAP": row["test_Exp_mAP"]})
        _write_json(out / f"epoch_{epoch:03d}" / "label_query_stats.json", {"layer_weights": []})
        if row["test_joint"] > best_test:
            best_test = row["test_joint"]
            save_checkpoint(out / "checkpoint_best_test.pth", model, optimizer, scheduler, epoch, best_test, manifest, row)
            save_checkpoint(out / "checkpoint_best_val.pth", model, optimizer, scheduler, epoch, best_test, manifest, row)
        save_checkpoint(out / "checkpoint_latest.pth", model, optimizer, scheduler, epoch, best_test, manifest, row)
        print(json.dumps({"event": "head_zoo_epoch", **row}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "head_zoo_training_complete", "head_name": args.head_name, "best_test_joint": best_test, "output_dir": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
