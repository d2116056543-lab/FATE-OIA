from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.bdd100k_scene_state_proxy import BDD100KSceneStateWeakLabelProvider
from fate_oia.engine.eval_egcaf_oia import evaluate_logits
from fate_oia.losses.egcaf_losses import EGCafLoss
from fate_oia.models.egcaf_oia_model import EGCafOIAModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.egcaf_artifacts import write_epoch_factor_artifacts, write_json


def _load_yaml_flat(path: str) -> dict[str, Any]:
    if not path:
        return {}
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
    return data if isinstance(data, dict) else {}


def _apply_defaults(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    cli = set(sys.argv[1:])
    for k, v in cfg.items():
        if hasattr(args, k) and f"--{k}" not in cli:
            setattr(args, k, v)
    if isinstance(getattr(args, "image_size", None), list):
        args.image_height, args.image_width = int(args.image_size[0]), int(args.image_size[1])
    if isinstance(getattr(args, "hook_layers", None), str):
        args.hook_layers = [int(x) for x in args.hook_layers.replace("[", "").replace("]", "").split(",") if x.strip()]
    return args


def _limited(ds, n: int):
    return Subset(ds, list(range(min(n, len(ds))))) if n and n > 0 else ds


def make_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    tfm = AspectRatioLetterboxTransform(args.image_height, args.image_width, patch_size=args.patch_size, return_meta=True)
    ds = BDDOIAMultiTaskDataset(args.data_root, args.raw_root, split=split, action_dim=args.action_dim, reason_dim=args.reason_dim, load_image=True, transform=tfm)
    max_n = args.max_train_samples if split == "train" else args.max_test_samples
    ds = _limited(ds, max_n)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    return batch["image"].to(device), batch["action"].float().to(device), batch["reason"].float().to(device), list(batch["file_name"])


@torch.no_grad()
def evaluate(model: EGCafOIAModel, loader: DataLoader, device: torch.device, out_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    model.eval()
    ac, af, ag, rr, ya, yr, names = [], [], [], [], [], [], []
    last_outputs: dict[str, Any] | None = None
    for batch in loader:
        images, y_action, y_reason, file_names = _batch_to_device(batch, device)
        outputs = model(images, return_artifacts=True, mode="test")
        ac.append(outputs["action_core_logits"].detach().cpu())
        af.append(outputs["action_final_logits"].detach().cpu())
        ag.append(outputs["guarded_action_logits"].detach().cpu())
        rr.append(outputs["reason_logits"].detach().cpu())
        ya.append(y_action.detach().cpu())
        yr.append(y_reason.detach().cpu())
        names.extend(file_names)
        if last_outputs is None:
            last_outputs = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in outputs.items() if k not in {"reason_attention", "reason_memory_tokens"}}
    tensors = {
        "action_core": torch.cat(ac),
        "action_final": torch.cat(af),
        "guarded_action": torch.cat(ag),
        "reason": torch.cat(rr),
        "labels_action": torch.cat(ya),
        "labels_reason": torch.cat(yr),
    }
    result = evaluate_logits(tensors["action_core"], tensors["action_final"], tensors["guarded_action"], tensors["reason"], tensors["labels_action"], tensors["labels_reason"])
    if out_dir:
        logit_dir = out_dir / "logits"
        logit_dir.mkdir(parents=True, exist_ok=True)
        for key, tensor in tensors.items():
            torch.save(tensor, logit_dir / f"{key}_test.pt")
        (logit_dir / "file_names_test.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    return result, tensors, last_outputs or {}


def _lr_for_epoch(base: float, epoch: int, epochs: int, warmup: int, min_lr: float) -> float:
    if epoch < warmup:
        return base * float(epoch + 1) / float(max(warmup, 1))
    t = (epoch - warmup) / max(epochs - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    provider = BDD100KSceneStateWeakLabelProvider(args.bdd100k_root)
    write_json(out / "run_manifest.json", {
        "repo": "FATE-OIA",
        "method": "EG-CAF-OIA V1.1",
        "command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "device": str(device),
        "direct_image_training": True,
        "no_feature_cache": bool(args.no_feature_cache),
        "test_only_eval": bool(args.test_only),
        "best_selection_split": "test",
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "pretrained_weights": args.pretrained_weights,
        "checkpoint_key": args.checkpoint_key,
        "bdd100k_scene_state_weak_labels": bool(args.bdd100k_root),
        "residual_enabled": bool(args.residual_enabled),
        "v1_1_patch_requirements": "enabled",
    })
    train_loader = make_loader(args, "train", True)
    test_loader = make_loader(args, "test", False)
    model = EGCafOIAModel(
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        hidden_dim=args.hidden_dim,
        pretrained_weights=args.pretrained_weights,
        checkpoint_key=args.checkpoint_key,
        patch_size=args.patch_size,
        hook_layers=args.hook_layers,
        lightweight_backbone=args.lightweight_backbone,
        residual_cap=args.residual_cap,
        residual_enabled=args.residual_enabled,
        sparse_method=args.sparse_method,
    ).to(device)
    criterion = EGCafLoss(
        rho=args.auxiliary_budget_rho,
        sufficiency_weight=args.sufficiency_weight,
        comprehensiveness_weight=args.comprehensiveness_weight,
        scene_state_weight=args.scene_state_weight,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))
    best = -1e9
    history = []
    for epoch in range(args.epochs):
        model.train()
        lr = _lr_for_epoch(args.lr, epoch, args.epochs, args.warmup_epochs, args.min_lr)
        _set_lr(opt, lr)
        running = []
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            images, y_action, y_reason, file_names = _batch_to_device(batch, device)
            scene_targets, scene_available = provider.batch(file_names, device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                outputs = model(images, bdd100k_scene_state=scene_targets, return_artifacts=(step == 1), mode="train")
                loss, stats = criterion(
                    outputs,
                    y_action,
                    y_reason,
                    shared_params=list(model.actor.parameters()) + list(model.reason_decoder.parameters()),
                    scene_state_targets=scene_targets,
                    scene_state_available=scene_available,
                )
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                # Real trainer-level reliability update from FactorJudge action-GT loss drops.
                judge_tensors = outputs.get("factor_judge_stats", {})
                per_action_drop = judge_tensors.get("selected_vs_random_action_loss_drop_per_action", torch.tensor(stats["selected_vs_random_action_loss_drop"], device=device))
                model.selector.update_reliability(
                    faith_delta=per_action_drop,
                    help_delta=torch.relu(per_action_drop),
                    hurt_delta=torch.relu(-per_action_drop),
                    selected_type_ids=outputs.get("selected_factor_types"),
                )
            running.append(stats)
            if step % args.print_every == 0:
                print(
                    f"epoch={epoch} batch={step}/{len(train_loader)} loss={stats['total_loss']:.4f} "
                    f"sel-vs-rand={stats['selected_vs_random_action_loss_drop']:.4f} "
                    f"lambda_exp={float(outputs['lambda_exp'].detach().mean().cpu()):.4f}",
                    flush=True,
                )
        epoch_dir = out / f"epoch_{epoch:03d}"
        metrics, _, eval_outputs = evaluate(model, test_loader, device, epoch_dir)
        latest_stats = running[-1] if running else {}
        eval_outputs["factor_judge_stats"] = outputs.get("factor_judge_stats", {})
        eval_outputs["gradient_budget_stats"] = {k: latest_stats.get(k) for k in ["norm_main", "norm_aux", "budget_scale", "rho", "used_true_grad_norm"]}
        write_epoch_factor_artifacts(epoch_dir, eval_outputs, metrics)
        write_json(epoch_dir / "loss_components.json", latest_stats)
        joint = float(metrics.get("joint_guarded", metrics.get("joint_final", -1e9)))
        row = {"epoch": epoch, "lr": lr, **metrics, **latest_stats}
        history.append(row)
        with (out / "metrics_summary.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out / "checkpoint_latest.pth")
        if joint > best:
            best = joint
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out / "checkpoint_best_test.pth")
            write_json(out / "metrics_best_test.json", row)
        print(
            f"EVAL epoch={epoch} joint_guarded={joint:.6f} "
            f"action_core_mF1={metrics.get('action_core_mF1', 0):.6f} "
            f"action_guarded_mF1={metrics.get('guarded_action_mF1', 0):.6f} "
            f"Exp_mF1={metrics.get('Exp_mF1', 0):.6f} Exp_mAP={metrics.get('Exp_mAP', 0):.6f}",
            flush=True,
        )
    write_json(out / "history.json", history)
    return {"output_dir": str(out), "best": best, "epochs": args.epochs}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--raw_root", default=".")
    ap.add_argument("--bdd100k_root", default=r"E:\sbw\BDD100K")
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--output_dir", default=r".background_runs\egcaf_oia_v1_smoke")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--image_size", nargs="*", type=int)
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--hook_layers", nargs="*", type=int, default=[3, 6, 9, 12])
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--lightweight_backbone", action="store_true")
    ap.add_argument("--residual_cap", type=float, default=0.03)
    ap.add_argument("--residual_enabled", action="store_true", default=False)
    ap.add_argument("--sparse_method", default="entmax15", choices=["entmax15", "sparsemax"])
    ap.add_argument("--auxiliary_budget_rho", type=float, default=0.10)
    ap.add_argument("--sufficiency_weight", type=float, default=0.10)
    ap.add_argument("--comprehensiveness_weight", type=float, default=0.10)
    ap.add_argument("--scene_state_weight", type=float, default=0.05)
    ap.add_argument("--print_every", type=int, default=200)
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    args = _apply_defaults(args, _load_yaml_flat(args.config))
    run(args)


if __name__ == "__main__":
    main()

