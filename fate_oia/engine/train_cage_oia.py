from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.engine.train_fate_oia import (
    apply_config_defaults,
    build_backbone,
    build_transform,
    extract_tokens,
    labels_from_batch,
    load_config_defaults,
    make_multilabel_criterion,
)
from fate_oia.losses.cage_losses import selected_vs_random_margin_loss
from fate_oia.models.cage_oia_model import CAGEOIAFeatureModel


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _limited(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        return Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def make_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    ds = BDDOIAMultiTaskDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        split=split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=True,
        transform=build_transform(args.image_height, args.image_width, args.patch_size, args.preserve_aspect_ratio, return_meta=True),
    )
    max_samples = args.max_train_samples if split == "train" else args.max_test_samples
    ds = _limited(ds, max_samples)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def _mask_token_indices(tokens: torch.Tensor, indices: torch.Tensor, fill: str = "mean") -> torch.Tensor:
    if indices.dim() != 2:
        raise ValueError("indices must be [B,K]")
    masked = tokens.clone()
    bsz, n_tokens, _ = tokens.shape
    mask = torch.zeros(bsz, n_tokens, dtype=torch.bool, device=tokens.device)
    mask.scatter_(1, indices.clamp_min(0).clamp_max(n_tokens - 1), True)
    if fill == "zero":
        masked[mask] = 0.0
    else:
        fill_value = tokens.mean(dim=1, keepdim=True).expand_as(tokens)
        masked[mask] = fill_value[mask]
    return masked


@torch.no_grad()
def selected_vs_random_action_drop(
    model: CAGEOIAFeatureModel,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    pred: dict[str, Any],
    action_dim: int,
    max_labels: int = 4,
    fill: str = "mean",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    topk = pred["evidence"].get("topk_indices")
    if topk is None:
        return rows
    label_count = min(action_dim, max_labels, topk.shape[1])
    rng = torch.Generator(device=tokens.device)
    rng.manual_seed(20260607)
    base_logits = pred["action_logits"][:, :action_dim]
    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(base_logits, labels[:, :action_dim], reduction="none")
    n_tokens = tokens.shape[1]
    for label_idx in range(label_count):
        selected_idx = topk[:, label_idx, :]
        k = int(selected_idx.shape[1])
        rand_idx = torch.stack([torch.randperm(n_tokens, device=tokens.device, generator=rng)[:k] for _ in range(tokens.shape[0])], dim=0)
        selected_tokens = _mask_token_indices(tokens, selected_idx, fill=fill)
        random_tokens = _mask_token_indices(tokens, rand_idx, fill=fill)
        selected_logits = model(selected_tokens)["action_logits"][:, :action_dim]
        random_logits = model(random_tokens)["action_logits"][:, :action_dim]
        selected_loss = torch.nn.functional.binary_cross_entropy_with_logits(selected_logits, labels[:, :action_dim], reduction="none")
        random_loss = torch.nn.functional.binary_cross_entropy_with_logits(random_logits, labels[:, :action_dim], reduction="none")
        sel_drop = selected_loss[:, label_idx] - base_loss[:, label_idx]
        rnd_drop = random_loss[:, label_idx] - base_loss[:, label_idx]
        rows.append(
            {
                "label_index": int(label_idx),
                "selected_drop": float(sel_drop.mean().detach().cpu()),
                "random_drop": float(rnd_drop.mean().detach().cpu()),
                "selected_minus_random": float((sel_drop - rnd_drop).mean().detach().cpu()),
                "available": True,
                "judge": "action_gt_loss_drop",
                "mask_fill": fill,
                "topk": k,
                "sample_count": int(tokens.shape[0]),
            }
        )
    return rows


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = CAGEOIAFeatureModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        evidence_topk=args.evidence_topk,
        transport_steps=args.transport_steps,
    )
    tokens = torch.randn(args.batch_size, args.num_tokens, args.input_dim)
    y_action = torch.randint(0, 2, (args.batch_size, args.action_dim)).float()
    y_reason = torch.randint(0, 2, (args.batch_size, args.reason_dim)).float()
    pred = model(tokens)
    action_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred["action_logits"], y_action)
    reason_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred["reason_logits"], y_reason)
    selected_drop = torch.zeros(args.batch_size, args.action_dim + args.reason_dim)
    random_drop = torch.zeros_like(selected_drop)
    positive_mask = torch.cat([y_action, y_reason], dim=1)
    evidence_loss = selected_vs_random_margin_loss(selected_drop, random_drop, positive_mask=positive_mask, margin=0.05)
    total = action_loss + reason_loss + 0.0 * evidence_loss
    total.backward()

    typed_shapes = {k: list(v.shape) for k, v in pred["transport"]["typed_edges"].items()}
    selected_schema = {
        "available": False,
        "reason": "smoke_schema_only_no_real_deletion_forward",
        "mode": "schema_smoke",
        "per_label": [
            {"label_index": i, "selected_drop": 0.0, "random_drop": 0.0, "selected_minus_random": 0.0}
            for i in range(args.action_dim + args.reason_dim)
        ],
    }
    summary = {
        "status": "PASS",
        "mode": "smoke_only",
        "input_shape": list(tokens.shape),
        "action_shape": list(pred["action_logits"].shape),
        "reason_shape": list(pred["reason_logits"].shape),
        "evidence_state_shape": list(pred["evidence"]["evidence_state"].shape),
        "evidence_scores_shape": list(pred["evidence"]["evidence_scores"].shape),
        "typed_edge_shapes": typed_shapes,
        "reason_reliability_shape": list(pred["reason_reliability"].shape),
        "action_gate_min": float(pred["action_gate"].min().detach()),
        "action_gate_max": float(pred["action_gate"].max().detach()),
        "loss": {"action": float(action_loss.detach()), "reason": float(reason_loss.detach())},
        "test_forward_uses_bdd100k_gt": False,
    }
    _write_json(out / "cage_smoke_summary.json", summary)
    _write_json(out / "selected_vs_random_by_label.json", selected_schema)
    _write_json(out / "run_manifest.json", vars(args))
    return summary


def _run_split(
    args: argparse.Namespace,
    backbone: nn.Module,
    model: CAGEOIAFeatureModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
) -> dict[str, Any]:
    model.train(train)
    total_loss = 0.0
    count = 0
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    base_logits_all: list[torch.Tensor] = []
    transport_logits_all: list[torch.Tensor] = []
    file_names: list[str] = []
    loss_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    accum = max(1, int(args.gradient_accumulation_steps))
    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for step, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = labels_from_batch(batch).to(device, non_blocking=True)
            with torch.no_grad():
                tokens = extract_tokens(backbone, images, args.n_last_blocks)
            pred = model(tokens)
            logits = torch.cat([pred["action_logits"], pred["reason_logits"]], dim=1)
            base_logits = torch.cat([pred["base_action_logits"], pred["base_reason_logits"]], dim=1)
            transport_logits = torch.cat([pred["transport_action_logits"], pred["transport_reason_logits"]], dim=1)
            main_loss = criterion(logits, labels)
            base_loss = criterion(base_logits, labels)
            transport_loss = criterion(transport_logits, labels)
            gate_penalty = pred["action_gate"].mean() * float(args.action_gate_l1)
            loss = main_loss + float(args.base_loss_weight) * base_loss + float(args.transport_loss_weight) * transport_loss + gate_penalty
            if train and optimizer is not None:
                (loss / float(accum)).backward()
                if ((step + 1) % accum == 0) or ((step + 1) == len(loader)):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            bs = images.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            count += bs
            logits_all.append(logits.detach().cpu())
            base_logits_all.append(base_logits.detach().cpu())
            transport_logits_all.append(transport_logits.detach().cpu())
            labels_all.append(labels.detach().cpu())
            fns = batch.get("file_name", [])
            if isinstance(fns, str):
                file_names.append(fns)
            else:
                file_names.extend([str(x) for x in fns])
            if len(token_rows) < args.max_saved_token_stats:
                token_rows.append(
                    {
                        "train": train,
                        "epoch": epoch,
                        "step": step,
                        "original_tokens": int(tokens.shape[1]),
                        "reduced_tokens": int(tokens.shape[1]),
                        "token_compression": "none",
                        "evidence_topk": int(args.evidence_topk),
                        "typed_transport_steps": int(args.transport_steps),
                        "action_gate_mean": float(pred["action_gate"].mean().detach().cpu()),
                        "reason_reliability_mean": float(pred["reason_reliability"].mean().detach().cpu()),
                        "evidence_confidence_mean": float(pred["evidence"]["evidence_confidence"].mean().detach().cpu()),
                    }
                )
            loss_rows.append(
                {
                    "train": train,
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "main_loss": float(main_loss.detach().cpu()),
                    "base_loss": float(base_loss.detach().cpu()),
                    "transport_loss": float(transport_loss.detach().cpu()),
                    "action_gate_l1": float(gate_penalty.detach().cpu()),
                    "lr": float(optimizer.param_groups[0]["lr"] if optimizer is not None else args.lr),
                    "effective_batch_size": int(args.batch_size * accum),
                    "loss_divided_by_accumulation": True,
                }
            )
            if (not train) and step < int(args.max_deletion_eval_batches):
                evidence_rows.extend(selected_vs_random_action_drop(model, tokens, labels, pred, args.action_dim, fill=args.deletion_mask_fill))
            if step % max(1, int(args.log_every)) == 0:
                print(
                    json.dumps(
                        {
                            "event": "cage_batch",
                            "epoch": epoch,
                            "train": train,
                            "step": step,
                            "batches": len(loader),
                            "loss": float(loss.detach().cpu()),
                            "main_loss": float(main_loss.detach().cpu()),
                            "base_loss": float(base_loss.detach().cpu()),
                            "transport_loss": float(transport_loss.detach().cpu()),
                            "action_gate_mean": float(pred["action_gate"].mean().detach().cpu()),
                            "reason_reliability_mean": float(pred["reason_reliability"].mean().detach().cpu()),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    logits_tensor = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0, args.action_dim + args.reason_dim)
    labels_tensor = torch.cat(labels_all, dim=0) if labels_all else torch.empty(0, args.action_dim + args.reason_dim)
    base_tensor = torch.cat(base_logits_all, dim=0) if base_logits_all else torch.empty(0, args.action_dim + args.reason_dim)
    transport_tensor = torch.cat(transport_logits_all, dim=0) if transport_logits_all else torch.empty(0, args.action_dim + args.reason_dim)
    metrics = evaluate_snna25(logits_tensor, labels_tensor, args.action_dim, threshold_mode=args.threshold_mode, fixed_threshold=args.eval_threshold)["metrics"] if labels_tensor.numel() else {}
    base_metrics = evaluate_snna25(base_tensor, labels_tensor, args.action_dim, threshold_mode=args.threshold_mode, fixed_threshold=args.eval_threshold)["metrics"] if labels_tensor.numel() else {}
    transport_metrics = evaluate_snna25(transport_tensor, labels_tensor, args.action_dim, threshold_mode=args.threshold_mode, fixed_threshold=args.eval_threshold)["metrics"] if labels_tensor.numel() else {}
    return {
        "loss": total_loss / max(count, 1),
        "count": count,
        "metrics": metrics,
        "branch_metrics": {"base": base_metrics, "transport": transport_metrics, "final": metrics},
        "logits": logits_tensor,
        "base_logits": base_tensor,
        "transport_logits": transport_tensor,
        "labels": labels_tensor,
        "file_names": file_names,
        "loss_components": loss_rows,
        "token_stats": token_rows,
        "selected_vs_random": evidence_rows,
    }


def _joint(stats: dict[str, Any]) -> float:
    m = stats.get("metrics", {})
    return 0.5 * float(m.get("Act_mF1", 0.0)) + 0.5 * float(m.get("Exp_mF1", 0.0))


def _save_epoch(out_dir: Path, epoch: int, train_stats: dict[str, Any], test_stats: dict[str, Any], args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    ep = out_dir / f"epoch_{epoch:03d}"
    ep.mkdir(parents=True, exist_ok=True)
    metrics = {
        "epoch": epoch,
        "test_loss": test_stats["loss"],
        "train_loss": train_stats["loss"],
        "joint": _joint(test_stats),
        **test_stats["metrics"],
        "base_branch": test_stats["branch_metrics"].get("base", {}),
        "transport_branch": test_stats["branch_metrics"].get("transport", {}),
    }
    _write_json(ep / "metrics_summary.json", metrics)
    _append_jsonl(out_dir / "metrics_summary.jsonl", metrics)
    _write_jsonl(ep / "loss_components.jsonl", train_stats["loss_components"] + test_stats["loss_components"])
    _write_jsonl(ep / "token_stats.jsonl", train_stats["token_stats"] + test_stats["token_stats"])
    _write_jsonl(ep / "selected_vs_random_by_label.jsonl", test_stats["selected_vs_random"])
    _write_json(ep / "run_manifest.json", manifest)
    torch.save(test_stats["logits"], ep / "logits_test.pt")
    torch.save(test_stats["base_logits"], ep / "logits_base_test.pt")
    torch.save(test_stats["transport_logits"], ep / "logits_transport_test.pt")
    torch.save(test_stats["labels"], ep / "labels_test.pt")
    _write_json(ep / "file_names_test.json", test_stats["file_names"])
    _write_json(out_dir / "metrics_latest.json", metrics)
    _write_jsonl(out_dir / "selected_vs_random_by_label.jsonl", test_stats["selected_vs_random"])


def _manifest(args: argparse.Namespace, out_dir: Path, train_count: int, test_count: int, embed_dim: int) -> dict[str, Any]:
    return {
        "repo_name": "FATE-OIA",
        "method": "CAGE-OIA-V1",
        "command": " ".join(sys.argv),
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "output_dir": str(out_dir),
        "data_root": str(args.data_root),
        "raw_root": str(args.raw_root),
        "train_split_count": int(train_count),
        "test_split_count": int(test_count),
        "eval_splits": ["test"],
        "best_selection_split": "test",
        "best_selection_metric": "joint_test_score",
        "best_selection_formula": "0.5 * Act_mF1 + 0.5 * Exp_mF1",
        "pretrained_weights": str(args.pretrained_weights),
        "checkpoint_key": str(args.checkpoint_key),
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "patch_size": int(args.patch_size),
        "backbone_embed_dim": int(embed_dim),
        "model_hidden_dim": int(args.hidden_dim),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(args.batch_size * max(1, args.gradient_accumulation_steps)),
        "lr": float(args.lr),
        "loss": str(args.loss),
        "loss_divided_by_accumulation": True,
        "uses_bdd100k_gt_in_test_forward": False,
        "config_resolved": vars(args),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAGE-OIA direct-image trainer and smoke entrypoint")
    parser.add_argument("--config", default="")
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--output_dir", default=".background_runs/cage_oia_v1_smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", default="E:/sbw/BDD-OIA/data")
    parser.add_argument("--raw_root", default="E:/sbw/BDD-OIA")
    parser.add_argument("--arch", default="vit_small")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--n_last_blocks", type=int, default=1)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--loss", choices=["asl", "bce"], default="asl")
    parser.add_argument("--asl_gamma_pos", type=float, default=0.0)
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument("--threshold_mode", choices=["fixed", "global", "per_label"], default="fixed")
    parser.add_argument("--eval_threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--action_dim", type=int, default=4)
    parser.add_argument("--reason_dim", type=int, default=21)
    parser.add_argument("--input_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_tokens", type=int, default=80)
    parser.add_argument("--evidence_topk", type=int, default=8)
    parser.add_argument("--transport_steps", type=int, default=2)
    parser.add_argument("--residual_cap", type=float, default=2.0)
    parser.add_argument("--base_loss_weight", type=float, default=0.15)
    parser.add_argument("--transport_loss_weight", type=float, default=0.05)
    parser.add_argument("--action_gate_l1", type=float, default=0.0)
    parser.add_argument("--max_saved_token_stats", type=int, default=64)
    parser.add_argument("--max_deletion_eval_batches", type=int, default=2)
    parser.add_argument("--deletion_mask_fill", choices=["mean", "zero"], default="mean")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config_defaults(args.config)
    apply_config_defaults(args, config)
    torch.manual_seed(args.seed)
    if args.smoke_only:
        summary = run_smoke(args)
        print(json.dumps(summary, indent=2), flush=True)
        return
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_loader = make_loader(args, "train", shuffle=True)
    test_loader = make_loader(args, "test", shuffle=False)
    backbone, embed_dim = build_backbone(args, device)
    model = CAGEOIAFeatureModel(
        input_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        evidence_topk=args.evidence_topk,
        transport_steps=args.transport_steps,
        residual_cap=args.residual_cap,
    ).to(device)
    criterion = make_multilabel_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    manifest = _manifest(args, out_dir, len(train_loader.dataset), len(test_loader.dataset), embed_dim)
    _write_json(out_dir / "run_manifest.json", manifest)
    best_joint = -math.inf
    for epoch in range(int(args.epochs)):
        train_stats = _run_split(args, backbone, model, train_loader, criterion, device, True, optimizer, epoch)
        test_stats = _run_split(args, backbone, model, test_loader, criterion, device, False, None, epoch)
        _save_epoch(out_dir, epoch, train_stats, test_stats, args, manifest)
        joint = _joint(test_stats)
        row = {"event": "cage_epoch", "epoch": epoch, "joint": joint, **test_stats["metrics"]}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args), "test_metrics": test_stats["metrics"], "joint": joint}
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        if joint > best_joint:
            best_joint = joint
            torch.save(ckpt, out_dir / "checkpoint_best_test.pth")
            _write_json(out_dir / "metrics_best_test.json", {"epoch": epoch, "joint": joint, **test_stats["metrics"]})


if __name__ == "__main__":
    main()
